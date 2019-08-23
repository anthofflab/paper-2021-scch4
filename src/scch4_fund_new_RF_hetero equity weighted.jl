function scch4_damages_fund_equity(base_temp::Array{Float64,2}, base_co2::Array{Float64,2}, pulse_temp::Array{Float64,2}, pulse_co2::Array{Float64,2})#, constant::Bool, discount_rate::Float64, elast::Float64, prtp::Float64)

           sneasy_years = collect(1765:2300)

           #Create vector of years DICE is run (note DICE2013 has 5 year timesteps)
           fund_years = collect(1950:2300)

           #Find indices in sneasy data that correspond to dice years
           #dice_index = findin(sneasy_years, dice_years)

           #Caluclate number of parameter samples from MCMC (equivalent to number of Sneasy model runs)
           #number_samples = size(base_temp, 2)

           #Extract climate output that corresponds to every 5th year for DICE timesteps.
           #sneasy_years  = sneasy_years[dice_index]
           #base_temp    = base_temp[dice_index, :]
           #pulse_temp      = pulse_temp[dice_index, :]

           #Caluclate number of parameter samples from MCMC (equivalent to number of Sneasy model runs)
           number_samples = size(base_temp, 1)

           #Indices to index into for temp/co2 to get out values for FUND periods
           sneasy_indices = findin(sneasy_years, fund_years)
           #Initialize arrays to store Sneasy results.
           #scch4 = zeros(number_samples)

           #Number of years to run DICE
           start_year    = 1950
           pulse_year = 2020
           end_year      = 2300

           #Calculate interpolated years for discounting.
           #interp_years = collect(start_year:end_year)
           pulse_index_fund = findin(fund_years, pulse_year)[1]

           # Number of years to run model.
           n_steps = length(start_year:end_year)

           #This is one year less than needed, but an extra year is added in construct fund because it does 1950:1950+nsteps for time period
           #Get two versions (base and marginal) of FUND model
           fund_base    = getfund(nsteps=n_steps-1)
           fund_pulse   = getfund(nsteps=n_steps-1)

           # Allocate arrays to hold regional damages.
           damage_base = zeros(n_steps, 16);
           damage_pulse= zeros(n_steps, 16);
           # These need to be 3-d arrays, row = year, column = region, layer = mcmc sample
           annual_marginal_damages = zeros(n_steps, 16, number_samples);
           pc_consumption= zeros(n_steps, 16, number_samples);
           baseline_population = zeros(n_steps, 16, number_samples);

           #Caluclate Social Cost of Methane for each parameter sample from MCMC.
           for i in 1:number_samples

               #FUND crashes with negative tempreature values. A couple chains give one or two years with temperatures that are -0.018.  Set these to 0.0.
               # Also set thees negative temperature values to 0.0 for DICE to maintina consistency across models.
               if minimum(base_temp[i,:]) <= 0.0 || minimum(pulse_temp[i,:]) <= 0.0
                   base_temp[i, find(base_temp[i,:] .< 0)] = 1e-15 #0.0
                   pulse_temp[i, find(pulse_temp[i,:] .< 0)] = 1e-15#0.0
               end

                   #Set parameters FUND needs to pick up from Sneasy output.
                #Temp parameters
                setparameter(fund_base, :climateregional, :inputtemp, base_temp[i, sneasy_indices])
                setparameter(fund_base, :biodiversity, :temp, base_temp[i, sneasy_indices])
                setparameter(fund_base, :ocean, :temp, base_temp[i, sneasy_indices])
                setparameter(fund_pulse, :climateregional, :inputtemp, pulse_temp[i, sneasy_indices])
                setparameter(fund_pulse, :biodiversity, :temp, pulse_temp[i, sneasy_indices])
                setparameter(fund_pulse, :ocean, :temp, pulse_temp[i, sneasy_indices])

                #CO2 Concnetration Parameters
                setparameter(fund_base, :impactagriculture, :acco2, base_co2[i, sneasy_indices])
                setparameter(fund_base, :impactextratropicalstorms, :acco2, base_co2[i, sneasy_indices])
                setparameter(fund_base, :impactforests, :acco2, base_co2[i, sneasy_indices])
                setparameter(fund_pulse, :impactagriculture, :acco2, pulse_co2[i, sneasy_indices])
                setparameter(fund_pulse, :impactextratropicalstorms, :acco2, pulse_co2[i, sneasy_indices])
                setparameter(fund_pulse, :impactforests, :acco2, pulse_co2[i, sneasy_indices])

               #Run two versions of DICE
               run(fund_base)
               run(fund_pulse)
               #   annual_marginal_damages[:,:,i] = (fund_pulse[:impactaggregation, :loss] .- fund_base[:impactaggregation, :loss]) ./ 10^6

                #Calculate marginal damages and scale from megatonne to tonne.
                # NOTE IT's ALREADY IN UNITS OF ONE TONNE
                annual_marginal_damages[:,:,i] = (fund_pulse[:impactaggregation, :loss] .- fund_base[:impactaggregation, :loss]) #./ 10^6

                # 1950 damage in FUND = NA (not 0.0), so just set to 0.0 here for convenience to avoid errors.
                annual_marginal_damages[1,:,i] = 0.0

                # Calculate per capita consumption by region
                pc_consumption[:,:,i] = fund_base[:socioeconomic,:ypc]

                # Calculate population (in individuals) for equity weighting
                baseline_population[:,:,i] = fund_base[:population, :population] .* 10^6

               # Take out growth effect of run 2 by transforming the damage from run 2 into % of GDP of run 2, and then
               # multiplying that with GDP of run 1
               #damage_marginal = dice_marginal[:damages,:DAMFRAC] .*dice_base[:grosseconomy,:YGROSS] * 10^12

               ##Convert from megatonnes to tonne
               #annual_marginal_damages= dice_interpolate((damage_marginal .- damage_base), start_year, end_year) ./ 10^6
               #annual_marginal_damages[:,i] = dice_interpolate((damage_pulse .- damage_base), 5) ./ 10^6
               #pc_consumption[:,i] = dice_interpolate(dice_base[:neteconomy, :CPC],5)
              
               #Print iteration number to track progress.
               #println("Completed marginal damage calculation ", i, " of ", number_samples, ".")

           end
           return annual_marginal_damages, pc_consumption, baseline_population
       end



function scch4_fund(annual_marginal_damages::Array{Float64,3}, pc_consumption::Array{Float64,3}, constant::Bool, discount_rate::Float64, elast::Float64, prtp::Float64)
    
    pulse_year = 2020
    # Calculate number of SC-CH4 values to calculate.
    n_samples = size(annual_marginal_damages,3)
    pulse_index_annual = findin(collect(1950:2300), 2020)[1]
    annual_years = collect(1950:2300)
    discounted_marginal_annual_damages_global = zeros(length(annual_years), n_samples)
    scch4_annual_share = zeros(length(annual_years), n_samples)
    cummulative_npv_damages = zeros(length(annual_years), n_samples)
    scch4 = zeros(n_samples)
    # Create array to hold discount factors
    df = zeros(length(annual_years), 16)

    #Calculate a constant discount rate.
    if constant
        for t=pulse_index_annual:length(annual_years)
            #Set discount timestep for 5 year intervals
            tt = annual_years[t]
            x = 1 / ((1. + discount_rate)^(tt-pulse_year))
            df[t,:] = x
        end
    end

    # Loop through each marginal damage estimate.
    for i = 1:n_samples
                #Should a constant discount rate be used?
#=       if constant == false
            for reg = 1:16
                x = 1.0
                for t=pulse_index_annual:length(annual_years)
                    df[t,reg] = x
                    gr = (pc_consumption[t,reg,i] - pc_consumption[t-1,reg,i]) / pc_consumption[t-1,reg,i]
                    x = x / (1. + prtp + elast * gr)
                end
            end
        end
=#       
       if constant == false
            #---------------------------------------------
            #If aggregating global consumption together.
            #----------------------------------------------
            pc_cons = sum(pc_consumption,2)
            for reg = 1:16
                x = 1.0
                for t=pulse_index_annual:length(annual_years)
                    df[t,reg] = x
                    gr = (pc_cons[t,1] - pc_cons[t-1,1]) / pc_cons[t-1,1]
                    x = x / (1. + prtp + elast * gr)
                end
            end
        end

        # Discount regional damages, and then sum them up to get npv global damages.
        discounted_marginal_annual_damages_global[2:end,i] = sum(annual_marginal_damages[2:end,:,i] .* df[2:end,:], 2)

        scch4[i] = sum(discounted_marginal_annual_damages_global[:,i])

        # Caluclate cumulative NPV damages and share of SC-CH4 achieved at each timestep.
        cummulative_npv_damages[1,i] = discounted_marginal_annual_damages_global[1,i]
        scch4_annual_share[1,i] = cummulative_npv_damages[1,i] / scch4[i]
        
        for t = 2:length(annual_years)
            cummulative_npv_damages[t,i] = cummulative_npv_damages[t-1,i] + discounted_marginal_annual_damages_global[t,i]
            scch4_annual_share[t,i] = cummulative_npv_damages[t,i] / scch4[i]
        end

        #println("Finsihed run ", i, " of ", n_samples)
    end
    return scch4, discounted_marginal_annual_damages_global, cummulative_npv_damages, scch4_annual_share
end



  for t=1:351
        temp_var = 0.
          for r=1:16
                temp_var += fund_population[t,r] * pc_consumption[t,r,i]^(1.0 - γ)
            end
            c_ede[t,i] = (temp_var / sum(fund_population[t,:]))^ (1.0/(1.0-γ))
        end
    end



    for t = 1:length(annual_years)
         cde_consumption[t,i] = sum(fund_population[t,:] ./ sum(fund_population[t,:]) .* pc_consumption[t,:,i] .^(1-γ)) ^ (1/(1-γ))
      end

   globc = 0.
    for r=1:regions
        globc += pop[1,r] * cpc[1,r]
    end
    
    cpc0x = focus_region == -1 ? globc / globpop[1] : cpc[1, focus_region]

    scc = 0.
    for t=1:timesteps, r=1:regions
        scc +=  md[t, r] *                      # P_rt * d_rt in the paper
                (1.0 + ρ)^(-(t-1)) *          # time discount factor
                (cpc0x / cpc[t, r])^γ *         # Normal equity weight
                (c_ede[t] / c_ede[1])^(γ - η) * # New thing
                (1.0 - I[t, r])^(-(γ + 1.0)) *  # New thing 2
                (1.0 - I[t, r])^(2 * α)         # This is Δ

#NEED CORRECT BASELINE POPULATION FOR EACH RUN (WILL DEPEND ON CLIMATE PARAMETER SAMPLE)

function scch4_fund_equity(annual_marginal_damages::Array{Float64,3}, pc_consumption::Array{Float64,3}, η::Float64, γ::Float64, ρ::Float64, fund_population::Array{Float64,3})
    
    pulse_year = 2020
    # Calculate number of SC-CH4 values to calculate.
    n_samples = size(annual_marginal_damages,3)
    pulse_index_annual = findin(collect(1950:2300), 2020)[1]
    annual_years = collect(1950:2300)
    cde_consumption = zeros(length(annual_years), n_samples)
    
    # Each column represents equity weighted SCC referenced to a different FUND region.
    scch4 = zeros(n_samples,16)

    # Loop through each marginal damage estimate.
    for i = 1:n_samples
               
      # Calculate cde conumption values (this does it for 1950:2300)
      for t = 1:length(annual_years)
         cde_consumption[t,i] = sum(fund_population[t,:,i] ./ sum(fund_population[t,:,i]) .* pc_consumption[t,:,i] .^(1-γ)) ^ (1/(1-γ))
      end
  
       # Calculate SCCs for each normalization region (this starts discounting at year of pulse, 2020)
       for r = 1:16
            for t = pulse_index_annual:351
                part1 = (cde_consumption[t,i] ./ cde_consumption[1,i]) .^ (γ-η)
                part2 = (pc_consumption[t,:,i] ./ pc_consumption[1,r,i]) .^ -γ
                scch4[i,r] += sum(part1 .* part2 .* (1+ρ) ^ (-t+pulse_index_annual) .* annual_marginal_damages[t,:,i])
            end
        end
        #println(i, " of ", n_samples)
    end

    return scch4
end




#-----------------------------------------------------------------------
#Calculate Damages for Broken Correlations with DICE for RCP 8.5
#-----------------------------------------------------------------------
#SET UP DIRECTORY TO SAVE INTO
model_name = "sneasy_magicc"
mcmc_file_name = "5_mill_cauchy_final_July6"

model_name = "sneasy_hector" 
mcmc_file_name = "RF_uncert_NEWrf_hetero_Dec7_wider_init_newch4iceinterp" #RF_uncert_NEWrf_hetero_Dec7_wider_init

model_name = "sneasy_fund"
mcmc_file_name = "RF_uncert_NEWrf_hetero_Dec9_wider_init_newch4iceinterp" #RF_uncert_NEWrf_hetero_Dec9_wider_init

model_name = "sneasy_fair"
mcmc_file_name = "NO_NATURAL_EMISS_RF_uncert_NEWrf_hetero_Dec12" #shorter version "NO_NATURAL_EMISS_RF_uncert_NEWrf"

model_scenario = "base_case"


rcp_scenario = "RCP85" #"RCP3PD"
output_directory = joinpath("scch4/fund/scch4_fund_results", model_name, rcp_scenario, mcmc_file_name)
#mkdir(output_directory)
output_directory = joinpath(output_directory, "equity_weighting_no_regional")
mkpath(output_directory)

using DataFrames
using Interpolations
using RCall
using CSVFiles

Conf_lvl_1 = 0.95
Conf_lvl_2 = 0.98

include("src/fund/src/fund_sneasyrun.jl")

# Load data (FUND needs temps and CO2)
base_temp = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/base_temp.csv"))));
marginal_temp = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/marginal_temp.csv"))));
base_co2 = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/base_co2.csv"))));
marginal_co2 = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/marginal_co2.csv"))));



# Calculate marginal damages (regional for FUND)
years_scch4 = collect(1950:2300)
#This wants temp array as [mcmc run, year] so it would be [10,000 x 536]
marg_damages, pc_cons, base_pop = scch4_damages_fund_equity(base_temp, base_co2, marginal_temp, marginal_co2)

# EQUITY WEIGHTED SC-CH4 for DIFFERENT ETA VALUES
scch4_equity_0_0 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.0, 0.015, base_pop)
scch4_equity_0_1 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.1, 0.015, base_pop)
scch4_equity_0_2 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.2, 0.015, base_pop)
scch4_equity_0_3 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.3, 0.015, base_pop)
scch4_equity_0_4 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.4, 0.015, base_pop)
scch4_equity_0_5 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.5, 0.015, base_pop)
scch4_equity_0_6 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.6, 0.015, base_pop)
scch4_equity_0_7 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.7, 0.015, base_pop)
scch4_equity_0_8 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.8, 0.015, base_pop)
scch4_equity_0_9 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.9, 0.015, base_pop)
scch4_equity_1_0 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.000001, 0.015, base_pop)
scch4_equity_1_1 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.1, 0.015, base_pop)
scch4_equity_1_2 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.2, 0.015, base_pop)
scch4_equity_1_3 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.3, 0.015, base_pop)
scch4_equity_1_4 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.4, 0.015, base_pop)
scch4_equity_1_5 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.5, 0.015, base_pop)

writecsv(joinpath(output_directory, "scch4_equity_eta_0_0.csv"), scch4_equity_0_0)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_1.csv"), scch4_equity_0_1)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_2.csv"), scch4_equity_0_2)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_3.csv"), scch4_equity_0_3)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_4.csv"), scch4_equity_0_4)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_5.csv"), scch4_equity_0_5)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_6.csv"), scch4_equity_0_6)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_7.csv"), scch4_equity_0_7)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_8.csv"), scch4_equity_0_8)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_9.csv"), scch4_equity_0_9)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_0.csv"), scch4_equity_1_0)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_1.csv"), scch4_equity_1_1)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_2.csv"), scch4_equity_1_2)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_3.csv"), scch4_equity_1_3)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_4.csv"), scch4_equity_1_4)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_5.csv"), scch4_equity_1_5)



equity_ci_reg[1,2], equity_ci_reg[1,3], equity_ci_reg[1,4], equity_ci_reg[1,5], equity_ci_reg[1,6] = CI_ordering(scch4_equity_1_5[:,1])

function CI_ordering(data)
  mean_d = mean(data)
  lower_95_d = quantile(data, 0.025)
  upper_95_d = quantile(data, 0.975)
  lower_98_d = quantile(data, 0.01)
  upper_98_d = quantile(data, 0.99)

  return mean_d, lower_95_d, upper_95_d, lower_98_d, upper_98_d
end

function equity_CI(region)
  eta=collect(0.0:0.1:1.5)

  x = DataFrame(eta=eta, mean=zeros(16), lower_95 = zeros(16), upper_95=zeros(16), lower_98=zeros(16), upper_98=zeros(16))
  x[1,2], x[1,3], x[1,4], x[1,5], x[1,6] = CI_ordering(scch4_equity_0_0[:,region])
  x[2,2], x[2,3], x[2,4], x[2,5], x[2,6] = CI_ordering(scch4_equity_0_1[:,region])
  x[3,2], x[3,3], x[3,4], x[3,5], x[3,6] = CI_ordering(scch4_equity_0_2[:,region])
  x[4,2], x[4,3], x[4,4], x[4,5], x[4,6] = CI_ordering(scch4_equity_0_3[:,region])
  x[5,2], x[5,3], x[5,4], x[5,5], x[5,6] = CI_ordering(scch4_equity_0_4[:,region])
  x[6,2], x[6,3], x[6,4], x[6,5], x[6,6] = CI_ordering(scch4_equity_0_5[:,region])
  x[7,2], x[7,3], x[7,4], x[7,5], x[7,6] = CI_ordering(scch4_equity_0_6[:,region])
  x[8,2], x[8,3], x[8,4], x[8,5], x[8,6] = CI_ordering(scch4_equity_0_7[:,region])
  x[9,2], x[9,3], x[9,4], x[9,5], x[9,6] = CI_ordering(scch4_equity_0_8[:,region])
  x[10,2], x[10,3], x[10,4], x[10,5], x[10,6] = CI_ordering(scch4_equity_0_9[:,region])
  x[11,2], x[11,3], x[11,4], x[11,5], x[11,6] = CI_ordering(scch4_equity_1_0[:,region])
  x[12,2], x[12,3], x[12,4], x[12,5], x[12,6] = CI_ordering(scch4_equity_1_1[:,region])
  x[13,2], x[13,3], x[13,4], x[13,5], x[13,6] = CI_ordering(scch4_equity_1_2[:,region])
  x[14,2], x[14,3], x[14,4], x[14,5], x[14,6] = CI_ordering(scch4_equity_1_3[:,region])
  x[15,2], x[15,3], x[15,4], x[15,5], x[15,6] = CI_ordering(scch4_equity_1_4[:,region])
  x[16,2], x[16,3], x[16,4], x[16,5], x[16,6] = CI_ordering(scch4_equity_1_5[:,region])

  return x
end

# CI GRAPHS FOR INDVUDAL REGIONS ACROSS DIFF ETA VALUES (FOR PLOTTING CONVENIENCE)

equity_CI_reg_1 = equity_CI(1)
equity_CI_reg_2 = equity_CI(2)
equity_CI_reg_3 = equity_CI(3)
equity_CI_reg_4 = equity_CI(4)
equity_CI_reg_5 = equity_CI(5)
equity_CI_reg_6 = equity_CI(6)
equity_CI_reg_7 = equity_CI(7)
equity_CI_reg_8 = equity_CI(8)
equity_CI_reg_9 = equity_CI(9)
equity_CI_reg_10 = equity_CI(10)
equity_CI_reg_11 = equity_CI(11)
equity_CI_reg_12 = equity_CI(12)
equity_CI_reg_13 = equity_CI(13)
equity_CI_reg_14 = equity_CI(14)
equity_CI_reg_15 = equity_CI(15)
equity_CI_reg_16 = equity_CI(16)

CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg1.csv"), equity_CI_reg_1)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg2.csv"), equity_CI_reg_2)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg3.csv"), equity_CI_reg_3)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg4.csv"), equity_CI_reg_4)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg5.csv"), equity_CI_reg_5)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg6.csv"), equity_CI_reg_6)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg7.csv"), equity_CI_reg_7)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg8.csv"), equity_CI_reg_8)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg9.csv"), equity_CI_reg_9)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg10.csv"), equity_CI_reg_10)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg11.csv"), equity_CI_reg_11)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg12.csv"), equity_CI_reg_12)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg13.csv"), equity_CI_reg_13)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg14.csv"), equity_CI_reg_14)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg15.csv"), equity_CI_reg_15)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg16.csv"), equity_CI_reg_16)

# SAVE
################################################################################
# BASE EQUITY WEIGHTING WITH RISK AVERSION
################################################################################
# EQUITY WEIGHTED SC-CH4 for DIFFERENT ETA VALUES
scch4_equity_0_0 = scch4_fund_equity(marg_damages, pc_cons, 0.0, 0.0, 0.01, base_pop)
scch4_equity_0_1 = scch4_fund_equity(marg_damages, pc_cons, 0.1, 0.1, 0.01, base_pop)
scch4_equity_0_2 = scch4_fund_equity(marg_damages, pc_cons, 0.2, 0.2, 0.01, base_pop)
scch4_equity_0_3 = scch4_fund_equity(marg_damages, pc_cons, 0.3, 0.3, 0.01, base_pop)
scch4_equity_0_4 = scch4_fund_equity(marg_damages, pc_cons, 0.4, 0.4, 0.01, base_pop)
scch4_equity_0_5 = scch4_fund_equity(marg_damages, pc_cons, 0.5, 0.5, 0.01, base_pop)
scch4_equity_0_6 = scch4_fund_equity(marg_damages, pc_cons, 0.6, 0.6, 0.01, base_pop)
scch4_equity_0_7 = scch4_fund_equity(marg_damages, pc_cons, 0.7, 0.7, 0.01, base_pop)
scch4_equity_0_8 = scch4_fund_equity(marg_damages, pc_cons, 0.8, 0.8, 0.01, base_pop)
scch4_equity_0_9 = scch4_fund_equity(marg_damages, pc_cons, 0.9, 0.9, 0.01, base_pop)
scch4_equity_1_0 = scch4_fund_equity(marg_damages, pc_cons, 1.000001, 1.000001, 0.01, base_pop)
scch4_equity_1_1 = scch4_fund_equity(marg_damages, pc_cons, 1.1, 1.1, 0.01, base_pop)
scch4_equity_1_2 = scch4_fund_equity(marg_damages, pc_cons, 1.2, 1.2, 0.01, base_pop)
scch4_equity_1_3 = scch4_fund_equity(marg_damages, pc_cons, 1.3, 1.3, 0.01, base_pop)
scch4_equity_1_4 = scch4_fund_equity(marg_damages, pc_cons, 1.4, 1.4, 0.01, base_pop)
scch4_equity_1_5 = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.5, 0.01, base_pop)
scch4_equity_1_6 = scch4_fund_equity(marg_damages, pc_cons, 1.6, 1.6, 0.01, base_pop)
scch4_equity_1_7 = scch4_fund_equity(marg_damages, pc_cons, 1.7, 1.7, 0.01, base_pop)
scch4_equity_1_8 = scch4_fund_equity(marg_damages, pc_cons, 1.8, 1.8, 0.01, base_pop)
scch4_equity_1_9 = scch4_fund_equity(marg_damages, pc_cons, 1.9, 1.9, 0.01, base_pop)
scch4_equity_2_0 = scch4_fund_equity(marg_damages, pc_cons, 2.0, 2.0, 0.01, base_pop)


writecsv(joinpath(output_directory, "scch4_equity_eta_0_0_constant.csv"), scch4_equity_0_0)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_1_constant.csv"), scch4_equity_0_1)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_2_constant.csv"), scch4_equity_0_2)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_3_constant.csv"), scch4_equity_0_3)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_4_constant.csv"), scch4_equity_0_4)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_5_constant.csv"), scch4_equity_0_5)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_6_constant.csv"), scch4_equity_0_6)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_7_constant.csv"), scch4_equity_0_7)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_8_constant.csv"), scch4_equity_0_8)
writecsv(joinpath(output_directory, "scch4_equity_eta_0_9_constant.csv"), scch4_equity_0_9)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_0_constant.csv"), scch4_equity_1_0)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_1_constant.csv"), scch4_equity_1_1)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_2_constant.csv"), scch4_equity_1_2)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_3_constant.csv"), scch4_equity_1_3)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_4_constant.csv"), scch4_equity_1_4)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_5_constant.csv"), scch4_equity_1_5)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_6_constant.csv"), scch4_equity_1_6)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_7_constant.csv"), scch4_equity_1_7)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_8_constant.csv"), scch4_equity_1_8)
writecsv(joinpath(output_directory, "scch4_equity_eta_1_9_constant.csv"), scch4_equity_1_9)
writecsv(joinpath(output_directory, "scch4_equity_eta_2_0_constant.csv"), scch4_equity_2_0)



# Read in data because i stupid and closed it
scch4_equity_0_0 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_0_constant.csv")
scch4_equity_0_1 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_1_constant.csv")
scch4_equity_0_2 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_2_constant.csv")
scch4_equity_0_3 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_3_constant.csv")
scch4_equity_0_4 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_4_constant.csv")
scch4_equity_0_5 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_5_constant.csv")
scch4_equity_0_6 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_6_constant.csv")
scch4_equity_0_7 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_7_constant.csv")
scch4_equity_0_8 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_8_constant.csv")
scch4_equity_0_9 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_0_9_constant.csv")
scch4_equity_1_0 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_0_constant.csv")
scch4_equity_1_1 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_1_constant.csv")
scch4_equity_1_2 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_2_constant.csv")
scch4_equity_1_3 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_3_constant.csv")
scch4_equity_1_4 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_4_constant.csv")
scch4_equity_1_5 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_5_constant.csv")
scch4_equity_1_6 = readcsv("scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional/scch4_equity_eta_1_6_constant.csv")

equity_ci_reg[1,2], equity_ci_reg[1,3], equity_ci_reg[1,4], equity_ci_reg[1,5], equity_ci_reg[1,6] = CI_ordering(scch4_equity_1_5[:,1])

function CI_ordering(data)
  mean_d = mean(data)
  lower_95_d = quantile(data, 0.025)
  upper_95_d = quantile(data, 0.975)
  lower_98_d = quantile(data, 0.01)
  upper_98_d = quantile(data, 0.99)

  return mean_d, lower_95_d, upper_95_d, lower_98_d, upper_98_d
end

function equity_CI(region)
  eta=collect(0.0:0.1:1.6)

  x = DataFrame(eta=eta, mean=zeros(17), lower_95 = zeros(17), upper_95=zeros(17), lower_98=zeros(17), upper_98=zeros(17))
  x[1,2], x[1,3], x[1,4], x[1,5], x[1,6] = CI_ordering(scch4_equity_0_0[:,region])
  x[2,2], x[2,3], x[2,4], x[2,5], x[2,6] = CI_ordering(scch4_equity_0_1[:,region])
  x[3,2], x[3,3], x[3,4], x[3,5], x[3,6] = CI_ordering(scch4_equity_0_2[:,region])
  x[4,2], x[4,3], x[4,4], x[4,5], x[4,6] = CI_ordering(scch4_equity_0_3[:,region])
  x[5,2], x[5,3], x[5,4], x[5,5], x[5,6] = CI_ordering(scch4_equity_0_4[:,region])
  x[6,2], x[6,3], x[6,4], x[6,5], x[6,6] = CI_ordering(scch4_equity_0_5[:,region])
  x[7,2], x[7,3], x[7,4], x[7,5], x[7,6] = CI_ordering(scch4_equity_0_6[:,region])
  x[8,2], x[8,3], x[8,4], x[8,5], x[8,6] = CI_ordering(scch4_equity_0_7[:,region])
  x[9,2], x[9,3], x[9,4], x[9,5], x[9,6] = CI_ordering(scch4_equity_0_8[:,region])
  x[10,2], x[10,3], x[10,4], x[10,5], x[10,6] = CI_ordering(scch4_equity_0_9[:,region])
  x[11,2], x[11,3], x[11,4], x[11,5], x[11,6] = CI_ordering(scch4_equity_1_0[:,region])
  x[12,2], x[12,3], x[12,4], x[12,5], x[12,6] = CI_ordering(scch4_equity_1_1[:,region])
  x[13,2], x[13,3], x[13,4], x[13,5], x[13,6] = CI_ordering(scch4_equity_1_2[:,region])
  x[14,2], x[14,3], x[14,4], x[14,5], x[14,6] = CI_ordering(scch4_equity_1_3[:,region])
  x[15,2], x[15,3], x[15,4], x[15,5], x[15,6] = CI_ordering(scch4_equity_1_4[:,region])
  x[16,2], x[16,3], x[16,4], x[16,5], x[16,6] = CI_ordering(scch4_equity_1_5[:,region])
  x[17,2], x[17,3], x[17,4], x[17,5], x[17,6] = CI_ordering(scch4_equity_1_6[:,region])

  return x
end

# CI GRAPHS FOR INDVUDAL REGIONS ACROSS DIFF ETA VALUES (FOR PLOTTING CONVENIENCE)

equity_CI_reg_1 = equity_CI(1)
equity_CI_reg_2 = equity_CI(2)
equity_CI_reg_3 = equity_CI(3)
equity_CI_reg_4 = equity_CI(4)
equity_CI_reg_5 = equity_CI(5)
equity_CI_reg_6 = equity_CI(6)
equity_CI_reg_7 = equity_CI(7)
equity_CI_reg_8 = equity_CI(8)
equity_CI_reg_9 = equity_CI(9)
equity_CI_reg_10 = equity_CI(10)
equity_CI_reg_11 = equity_CI(11)
equity_CI_reg_12 = equity_CI(12)
equity_CI_reg_13 = equity_CI(13)
equity_CI_reg_14 = equity_CI(14)
equity_CI_reg_15 = equity_CI(15)
equity_CI_reg_16 = equity_CI(16)

output_directory = "scch4/fund/scch4_fund_results/sneasy_magicc/RCP85/5_mill_cauchy_final_July6/equity_weighting_no_regional"

CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg1.csv"), equity_CI_reg_1)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg2.csv"), equity_CI_reg_2)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg3.csv"), equity_CI_reg_3)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg4.csv"), equity_CI_reg_4)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg5.csv"), equity_CI_reg_5)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg6.csv"), equity_CI_reg_6)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg7.csv"), equity_CI_reg_7)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg8.csv"), equity_CI_reg_8)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg9.csv"), equity_CI_reg_9)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg10.csv"), equity_CI_reg_10)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg11.csv"), equity_CI_reg_11)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg12.csv"), equity_CI_reg_12)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg13.csv"), equity_CI_reg_13)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg14.csv"), equity_CI_reg_14)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg15.csv"), equity_CI_reg_15)
CSVFiles.save(joinpath(output_directory, "scch4_equity_CI_reg16.csv"), equity_CI_reg_16)



















R"""
plot($eta, $ham[,6], type="l", ylim=c(0,3800))
lines($eta, $ham[,5])
lines($eta, $ham[,4])
lines($eta, $ham[,3])
lines($eta, $ham[,2])

lines($eta, $ham2[,2], col="red")
lines($eta, $ham2[,3], col="red")
lines($eta, $ham2[,4], col="red")
lines($eta, $ham2[,5], col="red")
lines($eta, $ham2[,6], col="red")
"""

scch4_equity_anthoff = scch4_fund_equity(marg_damages, pc_cons, 1.5, 0.7, 0.015, base_pop)
scch4_equity_double = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.4, 0.015, base_pop)
scch4_equity_princeton = scch4_fund_equity(marg_damages, pc_cons, 1.5, 1.5, 0.015, base_pop)


mean(scch4_anthoff,1)'

# 2.5% constant discount
scch4_25, annual_npv_damage_25, cumm_npv_damage_25, scch4_share_25 = scch4_fund(marg_damages, pc_cons, true, 0.025, 1.5, 0.015)
conf_cumm_dam_25 = confidence_int(years_scch4, cumm_npv_damage_25, Conf_lvl_1, Conf_lvl_2)
conf_share_25 = confidence_int(years_scch4, scch4_share_25, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_25 = confidence_int(years_scch4, annual_npv_damage_25, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_25.csv"), scch4_25)
writecsv(joinpath(output_directory, "annual_npv_damage_25.csv"), annual_npv_damage_25)
writecsv(joinpath(output_directory, "cumm_npv_damage_25.csv"), cumm_npv_damage_25)
writecsv(joinpath(output_directory, "scch4_share_25.csv"), scch4_share_25)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_25.csv"), conf_cumm_dam_25)
CSVFiles.save(joinpath(output_directory, "conf_share_25.csv"), conf_share_25)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_25.csv"), conf_annual_npv_dam_25)

# 3.0% constant discount
scch4_3, annual_npv_damage_3, cumm_npv_damage_3, scch4_share_3 = scch4_fund(marg_damages, pc_cons, true, 0.03, 1.5, 0.015)
conf_cumm_dam_3 = confidence_int(years_scch4, cumm_npv_damage_3, Conf_lvl_1, Conf_lvl_2)
conf_share_3 = confidence_int(years_scch4, scch4_share_3, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_3 = confidence_int(years_scch4, annual_npv_damage_3, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_3.csv"), scch4_3)
writecsv(joinpath(output_directory, "annual_npv_damage_3.csv"), annual_npv_damage_3)
writecsv(joinpath(output_directory, "cumm_npv_damage_3.csv"), cumm_npv_damage_3)
writecsv(joinpath(output_directory, "scch4_share_3.csv"), scch4_share_3)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_3.csv"), conf_cumm_dam_3)
CSVFiles.save(joinpath(output_directory, "conf_share_3.csv"), conf_share_3)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_3.csv"), conf_annual_npv_dam_3)


# 5.0% constant discount
scch4_5, annual_npv_damage_5, cumm_npv_damage_5, scch4_share_5 = scch4_fund(marg_damages, pc_cons, true, 0.05, 1.5, 0.015)
conf_cumm_dam_5 = confidence_int(years_scch4, cumm_npv_damage_5, Conf_lvl_1, Conf_lvl_2)
conf_share_5 = confidence_int(years_scch4, scch4_share_5, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_5 = confidence_int(years_scch4, annual_npv_damage_5, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_5.csv"), scch4_5)
writecsv(joinpath(output_directory, "annual_npv_damage_5.csv"), annual_npv_damage_5)
writecsv(joinpath(output_directory, "cumm_npv_damage_5.csv"), cumm_npv_damage_5)
writecsv(joinpath(output_directory, "scch4_share_5.csv"), scch4_share_5)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_5.csv"), conf_cumm_dam_5)
CSVFiles.save(joinpath(output_directory, "conf_share_5.csv"), conf_share_5)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_5.csv"), conf_annual_npv_dam_5)

# 7.0% constant discount
scch4_7, annual_npv_damage_7, cumm_npv_damage_7, scch4_share_7 = scch4_fund(marg_damages, pc_cons, true, 0.07, 1.5, 0.015)
conf_cumm_dam_7 = confidence_int(years_scch4, cumm_npv_damage_7, Conf_lvl_1, Conf_lvl_2)
conf_share_7 = confidence_int(years_scch4, scch4_share_7, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_7 = confidence_int(years_scch4, annual_npv_damage_7, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_7.csv"), scch4_7)
writecsv(joinpath(output_directory, "annual_npv_damage_7.csv"), annual_npv_damage_7)
writecsv(joinpath(output_directory, "cumm_npv_damage_7.csv"), cumm_npv_damage_7)
writecsv(joinpath(output_directory, "scch4_share_7.csv"), scch4_share_7)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_7.csv"), conf_cumm_dam_7)
CSVFiles.save(joinpath(output_directory, "conf_share_7.csv"), conf_share_7)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_7.csv"), conf_annual_npv_dam_7)

# Ramsey Nordhaus (eta = 1.5, prtp = 0.015)
scch4_nordhaus, annual_npv_damage_nordhaus, cumm_npv_damage_nordhaus, scch4_share_nordhaus = scch4_fund(marg_damages, pc_cons, false, 0.07, 1.5, 0.015)
conf_cumm_dam_nordhaus = confidence_int(years_scch4, cumm_npv_damage_nordhaus, Conf_lvl_1, Conf_lvl_2)
conf_share_nordhaus = confidence_int(years_scch4, scch4_share_nordhaus, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_nordhaus = confidence_int(years_scch4, annual_npv_damage_nordhaus, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_nordhaus.csv"), scch4_nordhaus)
writecsv(joinpath(output_directory, "annual_npv_damage_nordhaus.csv"), annual_npv_damage_nordhaus)
writecsv(joinpath(output_directory, "cumm_npv_damage_nordhaus.csv"), cumm_npv_damage_nordhaus)
writecsv(joinpath(output_directory, "scch4_share_nordhaus.csv"), scch4_share_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_nordhaus.csv"), conf_cumm_dam_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_share_nordhaus.csv"), conf_share_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_nordhaus.csv"), conf_annual_npv_dam_nordhaus)

# Ramsey Sterm (eta = 1.0, prtp = 0.001)
scch4_stern, annual_npv_damage_stern, cumm_npv_damage_stern, scch4_share_stern = scch4_fund(marg_damages, pc_cons, false, 0.07, 1.0, 0.001)
conf_cumm_dam_stern = confidence_int(years_scch4, cumm_npv_damage_stern, Conf_lvl_1, Conf_lvl_2)
conf_share_stern = confidence_int(years_scch4, scch4_share_stern, Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_stern = confidence_int(years_scch4, annual_npv_damage_stern, Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_stern_GLOBAL_CPC_GROWTH.csv"), scch4_stern)
writecsv(joinpath(output_directory, "annual_npv_damage_stern_GLOBAL_CPC_GROWTH.csv"), annual_npv_damage_stern)
writecsv(joinpath(output_directory, "cumm_npv_damage_stern_GLOBAL_CPC_GROWTH.csv"), cumm_npv_damage_stern)
writecsv(joinpath(output_directory, "scch4_share_stern_GLOBAL_CPC_GROWTH.csv"), scch4_share_stern)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_stern_GLOBAL_CPC_GROWTH.csv"), conf_cumm_dam_stern)
CSVFiles.save(joinpath(output_directory, "conf_share_stern_GLOBAL_CPC_GROWTH.csv"), conf_share_stern)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_stern_GLOBAL_CPC_GROWTH.csv"), conf_annual_npv_dam_stern)


spaghetti_damages_double(scch4_share_stern, scch4_share_nordhaus, conf_share_stern, conf_share_nordhaus, "dodgerblue", "red", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1.05], "Year", "Share", "Test")

R"""
plot(1950:2300, $conf_share_25[,2], type="l", ylim=c(0,1), xlim=c(2020,2150), col="red")
lines(1950:2300, $conf_share_3[,2], col="darkorange")
lines(1950:2300, $conf_share_5[,2], col="forestgreen")
lines(1950:2300, $conf_share_7[,2], col="blue")
lines(1950:2300, $conf_share_nordhaus[,2], col="magenta", lty=2)
lines(1950:2300, $conf_share_stern[,2], col="black", lty=2)
"""

R"""
plot(1950:2300, $conf_annual_npv_dam_25[,2], type="l", col="red", xlim=c(2020,2150))
lines(1950:2300, $conf_annual_npv_dam_3[,2], col="darkorange")
lines(1950:2300, $conf_annual_npv_dam_5[,2], col="forestgreen")
lines(1950:2300, $conf_annual_npv_dam_7[,2], col="blue")
lines(1950:2300, $conf_annual_npv_dam_nordhaus[,2], col="magenta", lty=2)
lines(1950:2300, $conf_annual_npv_dam_stern[,2], col="black", lty=2)
"""



R"""
plot(2010:2300, $scch4_share[,1], type="l", col="red", xlim=c(2020,2100))
lines(2010:2300, $scch4_share[,2], col="red")
lines(2010:2300, $scch4_share[,3], col="red")
lines(2010:2300, $scch4_share[,4], col="red")
lines(2010:2300, $scch4_share[,5], col="red")
lines(2010:2300, $scch4_share2[,1], col="blue")
lines(2010:2300, $scch4_share2[,2], col="blue")
lines(2010:2300, $scch4_share2[,3], col="blue")
lines(2010:2300, $scch4_share2[,4], col="blue")
lines(2010:2300, $scch4_share2[,5], col="blue")
lines(2010:2300, $scch4_share3[,1], col="forestgreen")
lines(2010:2300, $scch4_share3[,2], col="forestgreen")
lines(2010:2300, $scch4_share3[,3], col="forestgreen")
lines(2010:2300, $scch4_share3[,4], col="forestgreen")
lines(2010:2300, $scch4_share3[,5], col="forestgreen")
lines(2010:2300, $scch4_share4[,1], col="magenta")
lines(2010:2300, $scch4_share4[,2], col="magenta")
lines(2010:2300, $scch4_share4[,3], col="magenta")
lines(2010:2300, $scch4_share4[,4], col="magenta")
lines(2010:2300, $scch4_share4[,5], col="magenta")
"""


function spaghetti_damages_double(data1, data2, conf_data1, conf_data2, color1, color2, alpha, start_year, end_year, cutoff_year, num_sims, x_range, y_range, x_title, y_title, main_title)

    R"""
    start_year = $start_year
    end_year = $end_year
    cutoff_year = $cutoff_year
    data1 = $data1
    data2 = $data2
    conf_data1 = $conf_data1
    conf_data2 = $conf_data2
    color1 = $color1
    color2 = $color2
    alpha = $alpha
    num_sims = $num_sims
    x_range = c($x_range[1], $x_range[2])
    y_range = c($y_range[1], $y_range[2])
    x_title = $x_title
    y_title = $y_title
    main_title = $main_title

    start_index = which(2010:2300==start_year)
    end_index = which(2010:2300==end_year)

    #Select 1000 values if num_sims is > 1000 (otherwise line overlap causes plot to become a solid color).
    rand_indices = sample(1:ncol(data1), num_sims)
    reshaped_data1 = melt(data1[start_index:end_index,rand_indices])
    reshaped_data2 = melt(data2[start_index:end_index,rand_indices])
    plot_data = data.frame(Period = rep(start_year:end_year, times=num_sims), reshaped_data1, reshaped_data2)
    colnames(plot_data) = c("Year", "Count1", "Model_Run1", "Value1", "Count2", "Model_Run2", "Value2")
    #mean_data = data.frame(Period = ($start_year:$end_year), mean = $mean_data)

    p = ggplot()
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value1", group="Model_Run1"), colour=color1, alpha=alpha)

    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.1)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.7)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.7)

    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value2", group="Model_Run2"), colour=color2, alpha=alpha)
    p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour=color2, size=1.5)
    p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour=color1, size=1.5)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour="black", size=0.8)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.5)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.5)

    #p = p + geom_vline(xintercept=cutoff_year, linetype="dashed", lwd=1.2, colour="red")

    p = p + ylim(y_range)
    p = p + xlim(x_range)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle(main_title)
    p = p + theme(panel.background = element_rect(fill = "transparent",colour = "black"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(),
                  legend.position="none",
                  plot.title = element_text(face="bold", size=13, hjust = 0.5),
                  axis.text = element_text(size=12, colour="black"),
                  axis.title = element_text(size=12))
    """
end

function spaghetti_damages_quad(data1, data2, data3, data4, conf_data1, conf_data2, conf_data3, conf_data4, color1, color2, color3, color4, alpha, start_year, end_year, cutoff_year, num_sims, x_range, y_range, x_title, y_title, main_title)

    R"""
    start_year = $start_year
    end_year = $end_year
    cutoff_year = $cutoff_year
    data1 = $data1
    data2 = $data2
    data3 = $data3
    data4 = $data4
    conf_data1 = $conf_data1
    conf_data2 = $conf_data2
    conf_data3 = $conf_data3
    conf_data4 = $conf_data4
    color1 = $color1
    color2 = $color2
    color3 = $color3
    color4 = $color4
    alpha = $alpha
    num_sims = $num_sims
    x_range = c($x_range[1], $x_range[2])
    y_range = c($y_range[1], $y_range[2])
    x_title = $x_title
    y_title = $y_title
    main_title = $main_title

    start_index = which(2010:2300==start_year)
    end_index = which(2010:2300==end_year)

    #Select 1000 values if num_sims is > 1000 (otherwise line overlap causes plot to become a solid color).
    rand_indices = sample(1:ncol(data1), num_sims)
    reshaped_data1 = melt(data1[start_index:end_index,rand_indices])
    reshaped_data2 = melt(data2[start_index:end_index,rand_indices])
    reshaped_data3 = melt(data3[start_index:end_index,rand_indices])
    reshaped_data4 = melt(data4[start_index:end_index,rand_indices])
    plot_data = data.frame(Period = rep(start_year:end_year, times=num_sims), reshaped_data1, reshaped_data2, reshaped_data3, reshaped_data4)
    colnames(plot_data) = c("Year", "Count1", "Model_Run1", "Value1", "Count2", "Model_Run2", "Value2", "Count3", "Model_Run3", "Value3","Count4", "Model_Run4", "Value4")
    #mean_data = data.frame(Period = ($start_year:$end_year), mean = $mean_data)

    p = ggplot()
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value1", group="Model_Run1"), colour=color1, alpha=alpha)

    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.1)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.7)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.7)

    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value2", group="Model_Run2"), colour=color2, alpha=alpha)
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value3", group="Model_Run3"), colour=color3, alpha=alpha)
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value4", group="Model_Run4"), colour=color4, alpha=alpha)

    p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.2)
    p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.2)
    p = p + geom_line(data=conf_data3, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.2)
    p = p + geom_line(data=conf_data4, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.2)

    p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour=color1, size=0.8)
    p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour=color2, size=0.8)
    p = p + geom_line(data=conf_data3, aes_string(x="Year", y="Mean_Chain"), colour=color3, size=0.8)
    p = p + geom_line(data=conf_data4, aes_string(x="Year", y="Mean_Chain"), colour=color4, size=0.8)

    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour="black", size=0.8)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.5)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.5)

    #p = p + geom_vline(xintercept=cutoff_year, linetype="dashed", lwd=1.2, colour="red")

    p = p + ylim(y_range)
    p = p + xlim(x_range)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle(main_title)
    p = p + theme(panel.background = element_rect(fill = "transparent",colour = "black"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(),
                  legend.position="none",
                  plot.title = element_text(face="bold", size=13, hjust = 0.5),
                  axis.text = element_text(size=12, colour="black"),
                  axis.title = element_text(size=12))
    """
end



function spaghetti_damages_five(data1, data2, data3, data4, data5, conf_data1, conf_data2, conf_data3, conf_data4, conf_data5, conf_data_nord, color1, color2, color3, color4, color5, alpha, start_year, end_year, cutoff_year, num_sims, x_range, y_range, x_title, y_title, main_title)

    R"""
    start_year = $start_year
    end_year = $end_year
    cutoff_year = $cutoff_year
    data1 = $data1
    data2 = $data2
    data3 = $data3
    data4 = $data4
    data5 = $data5
    conf_data1 = $conf_data1
    conf_data2 = $conf_data2
    conf_data3 = $conf_data3
    conf_data4 = $conf_data4
    conf_data5 = $conf_data5
    conf_data_nord = $conf_data_nord
    color1 = $color1
    color2 = $color2
    color3 = $color3
    color4 = $color4
    color5 = $color5
    alpha = $alpha
    num_sims = $num_sims
    x_range = c($x_range[1], $x_range[2])
    y_range = c($y_range[1], $y_range[2])
    x_title = $x_title
    y_title = $y_title
    main_title = $main_title

    start_index = which(2010:2300==start_year)
    end_index = which(2010:2300==end_year)

    #Select 1000 values if num_sims is > 1000 (otherwise line overlap causes plot to become a solid color).
    rand_indices = sample(1:ncol(data1), num_sims)
    reshaped_data1 = melt(data1[start_index:end_index,rand_indices])
    reshaped_data2 = melt(data2[start_index:end_index,rand_indices])
    reshaped_data3 = melt(data3[start_index:end_index,rand_indices])
    reshaped_data4 = melt(data4[start_index:end_index,rand_indices])
    reshaped_data5 = melt(data5[start_index:end_index,rand_indices])
    plot_data = data.frame(Period = rep(start_year:end_year, times=num_sims), reshaped_data1, reshaped_data2, reshaped_data3, reshaped_data4, reshaped_data5)
    colnames(plot_data) = c("Year", "Count1", "Model_Run1", "Value1", "Count2", "Model_Run2", "Value2", "Count3", "Model_Run3", "Value3","Count4", "Model_Run4", "Value4", "Count5", "Model_Run5", "Value5")
    #mean_data = data.frame(Period = ($start_year:$end_year), mean = $mean_data)

    p = ggplot()
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value1", group="Model_Run1"), colour=color1, alpha=alpha)

    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.1)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.7)
    #p = p + geom_line(data=conf_data1, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.7)

    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value2", group="Model_Run2"), colour=color2, alpha=alpha)
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value3", group="Model_Run3"), colour=color3, alpha=alpha)
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value4", group="Model_Run4"), colour=color4, alpha=alpha)
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value4", group="Model_Run5"), colour=color5, alpha=alpha)

    p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.4)
    p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.4)
    p = p + geom_line(data=conf_data3, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.4)
    p = p + geom_line(data=conf_data4, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.4)
    p = p + geom_line(data=conf_data5, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.4)

    p = p + geom_line(data=conf_data1, aes_string(x="Year", y="Mean_Chain"), colour=color1, size=0.8)
    p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour=color2, size=0.8)
    p = p + geom_line(data=conf_data3, aes_string(x="Year", y="Mean_Chain"), colour=color3, size=0.8)
    p = p + geom_line(data=conf_data4, aes_string(x="Year", y="Mean_Chain"), colour=color4, size=0.8)
    p = p + geom_line(data=conf_data5, aes_string(x="Year", y="Mean_Chain"), colour="black", size=0.8)
    p = p + geom_line(data=conf_data_nord, aes_string(x="Year", y="Mean_Chain"), colour=color5, size=0.8, linetype="dashed")

    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="Mean_Chain"), colour="black", size=0.8)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.5)
    #p = p + geom_line(data=conf_data2, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.5)

    #p = p + geom_vline(xintercept=cutoff_year, linetype="dashed", lwd=1.2, colour="red")

    p = p + ylim(y_range)
    p = p + xlim(x_range)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle(main_title)
    p = p + theme(panel.background = element_rect(fill = "transparent",colour = "black"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(),
                  legend.position="none",
                  plot.title = element_text(face="bold", size=13, hjust = 0.5),
                  axis.text = element_text(size=12, colour="black"),
                  axis.title = element_text(size=12))
    """
end


scch4_8model = function(dice, fund, alpha_d1, alpha_d2, alpha_d3, alpha_d4, alpha_f1, alpha_f2, alpha_f3, alpha_f4, size, x_range, y_range, x_title, y_title, main_title){

    p = ggplot()

    #DICE
    p = p + geom_density(data=dice, aes_string(x="scch4_oldrf"), fill="red",   alpha=alpha_d1, size=size,linetype="dashed")
    p = p + geom_density(data=dice, aes_string(x="scch4_base"), fill="darkorange",   alpha=alpha_d2, size=size,linetype="dashed")
    p = p + geom_density(data=dice, aes_string(x="scch4_broken"), fill="forestgreen",   alpha=alpha_d3, size=size,linetype="dashed")
    p = p + geom_density(data=dice, aes_string(x="scch4_roebaker"), fill="dodgerblue",      alpha=alpha_d4, size=size,linetype="dashed")

    p = p + geom_density(data=fund, aes_string(x="scch4_oldrf"), fill="red",  alpha=alpha_f1, size=size)
    p = p + geom_density(data=fund, aes_string(x="scch4_base"), fill="darkorange",  alpha=alpha_f2, size=size)
    p = p + geom_density(data=fund, aes_string(x="scch4_broken"), fill="forestgreen",  alpha=alpha_f3, size=size)
    p = p + geom_density(data=fund, aes_string(x="scch4_roebaker"), fill="dodgerblue",     alpha=alpha_f4, size=size)

    p = p + xlim(x_range)
    p = p + ylim(y_range)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle(main_title)
    p = p + theme(panel.background = element_rect(fill = "transparent",colour = "black"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(),
                  legend.position="none",
                  plot.title = element_text(face="bold", size=13, hjust = 0.5),
                  axis.text = element_text(size=12, colour="black"),
                  axis.title = element_text(size=12))
}
#=

scch4_8model = function(dice, fund, alpha_d1, alpha_d2, alpha_d3, alpha_d4, alpha_f1, alpha_f2, alpha_f3, alpha_f4, size, x_range, y_range, x_title, y_title, main_title){

    p = ggplot()

    #DICE
    p = p + geom_density(data=dice, aes_string(x="scch4_oldrf"), fill="darkorange",  colour="darkorange", alpha=alpha_d1, size=size)
        p = p + geom_density(data=dice, aes_string(x="scch4_base"), fill="red",  colour="red", alpha=alpha_d2, size=size)
    p = p + geom_density(data=dice, aes_string(x="scch4_broken"), fill="dodgerblue",  colour="dodgerblue", alpha=alpha_d3, size=size)
    p = p + geom_density(data=dice, aes_string(x="scch4_roebaker"), fill="orchid4",     colour="orchid4", alpha=alpha_d4, size=size)

    p = p + geom_density(data=fund, aes_string(x="scch4_oldrf"), fill="darkorange",  colour="darkorange",alpha=alpha_f1, size=size)
        p = p + geom_density(data=fund, aes_string(x="scch4_base"), fill="red",  colour="red",alpha=alpha_f2, size=size)
    p = p + geom_density(data=fund, aes_string(x="scch4_broken"), fill="dodgerblue",  colour="dodgerblue",alpha=alpha_f3, size=size)
    p = p + geom_density(data=fund, aes_string(x="scch4_roebaker"), fill="orchid4",     colour="orchid4",alpha=alpha_f4, size=size)

    p = p + xlim(x_range)
    p = p + ylim(y_range)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle(main_title)
    p = p + theme(panel.background = element_rect(fill = "transparent",colour = "black"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(),
                  legend.position="none",
                  plot.title = element_text(face="bold", size=13, hjust = 0.5),
                  axis.text = element_text(size=12, colour="black"),
                  axis.title = element_text(size=12))
}
=#

three_data_dice = data.frame(three_fair, three_fund, three_hector, three_magicc)
three_data_fund = data.frame(three_fund_fair, three_fund_fund, three_fund_hector, three_fund_magicc)
colnames(three_data_dice) = c("scch4_base", "scch4_oldrf", "scch4_roebaker", "scch4_broken")
colnames(three_data_fund) = c("scch4_base", "scch4_oldrf", "scch4_roebaker", "scch4_broken")

ham_3 = scch4_8model(three_data_dice, three_data_fund, 0.7, 0.6, 0.6, 0.6, 0.7, 0.6, 0.6, 0.6, 1.0, c(-100,2200), c(0,0.0042), "SCCH4", "pdf", "HAM")
ham3_pt = ham_3 + geom_point(aes(x=c(846,951,1049,995, 265,295,329,310), y=c(0,0,0,0,0,0,0,0)), size=3, shape=c(23,23,23,23,21,21,21,21), stroke=1.2, fill=c("red", "darkorange", "forestgreen", "dodgerblue", "red", "darkorange","forestgreen", "dodgerblue"))
ham4_pt = ham3_pt + geom_vline(xintercept=1100, linetype="dashed", lwd=0.4, colour="blue")

png(filename = "RCP85_all_models_main_result_EPALINE5.png", type="cairo", units="in", width=9, height=5, res=200)
print(ham4_pt)
dev.off()


dice_data = data.frame(dice_base, dice_oldRF, dice_roebaker, dice_broken)
colnames(dice_data) = c("scch4_base", "scch4_oldrf", "scch4_roebaker", "scch4_broken")

fund_data = data.frame(fund_base, fund_oldRF, fund_roebaker, fund_corr)
colnames(fund_data) = c("scch4_base", "scch4_oldrf", "scch4_roebaker", "scch4_broken")

ham = scch4_8model(dice_data, fund_data, 0.7, 0.6, 0.5, 0.5, 0.7, 0.6, 0.5, 0.5, 1.0, c(-100,4100), c(0,0.0045), "SCCH4", "pdf", "HAM")


png(filename = "RCP85_magicc_scch4_scenarios.png", type="cairo", units="in", width=9, height=5, res=200)
print(ham)
dev.off()

#Mean points
ham2 = ham + geom_point(aes(x=c(278, 329, 343, 465, 895, 1049, 1101, 1541), y=c(0,0,0,0,0,0,0,0)), size=1)
#95th percentile
ham2 = ham + geom_point(aes(x=c(470, 565, 710, 1117, 1494, 1768, 2189, 4017), y=c(0,0,0,0,0,0,0,0)), size=4, shape =c(19,19,19,19,15,15,15,15))
ham2 = ham2 + geom_point(aes(x=c(470, 565, 710, 1117, 1494, 1768, 2189, 4017), y=c(0,0,0,0,0,0,0,0)), size=2, shape =c(19,19,19,19,15,15,15,15), colour=c("red", "darkorange", "forestgreen", "dodgerblue","red", "darkorange", "forestgreen", "dodgerblue"))


#95th percentile (97.5 upper range)
ham2 = ham + geom_point(aes(x=c(470, 565, 710, 1117, 1494, 1768, 2189, 4017), y=c(0,0,0,0,0,0,0,0)), size=4, shape =c(21,21,21,21,23,23,23,23), fill=c("red", "darkorange", "forestgreen", "dodgerblue","red", "darkorange", "forestgreen", "dodgerblue"), stroke=1.2)
ham2 = ham2 + geom_point(aes(x=c(470, 565, 710, 1117, 1494, 1768, 2189, 4017), y=c(0,0,0,0,0,0,0,0)), size=2, shape =c(21,21,21,21,23,23,23,23), fill="black")
#95th percentile (just acutal 95% in quantile function)
ham2 = ham + geom_point(aes(x=c(435, 519, 642, 1012, 1358, 1610, 1973, 3541), y=c(0,0,0,0,0,0,0,0)), size=3, shape =c(21,21,21,21,23,23,23,23), fill=c("red", "darkorange", "forestgreen", "dodgerblue","red", "darkorange", "forestgreen", "dodgerblue"), stroke=1.0)
# Add vertical line for EPA DICE+FUND average 95% value.
ham2 = ham2 + geom_vline(xintercept=2310, linetype="dashed", lwd=0.4, colour="blue")


png(filename = "magicc85scenario_EPA_95CI.png", type="cairo", units="in", width=9, height=5, res=200)
print(ham2)
dev.off()

