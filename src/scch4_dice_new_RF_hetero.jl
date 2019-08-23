function scch4_damages_dice(base_temp::Array{Float64,2}, pulse_temp::Array{Float64,2})#, constant::Bool, discount_rate::Float64, elast::Float64, prtp::Float64)

           sneasy_years = collect(1765:2300)

           #Create vector of years DICE is run (note DICE2013 has 5 year timesteps)
           dice_years = collect(2010:5:2300)

           #Find indices in sneasy data that correspond to dice years
           dice_index = findin(sneasy_years, dice_years)

           #Extract climate output that corresponds to every 5th year for DICE timesteps.
           sneasy_years  = sneasy_years[dice_index]
           base_temp    = base_temp[:, dice_index]
           pulse_temp      = pulse_temp[:, dice_index]

           #Caluclate number of parameter samples from MCMC (equivalent to number of Sneasy model runs)
           number_samples = size(base_temp, 1)

           #Initialize arrays to store Sneasy results.
           #scch4 = zeros(number_samples)

           #Number of years to run DICE
           start_year    = 2010
           pulse_year = 2020
           end_year      = 2300

           #Calculate interpolated years for discounting.
           #interp_years = collect(start_year:end_year)
           annual_years = collect(start_year:end_year)
           pulse_index_annual = findin(annual_years, pulse_year)[1]

           #Get two versions (base and marginal) of DICE model
           dice_base    = getdice()
           dice_pulse   = getdice()

           damage_base = zeros(length(dice_years))
           damage_pulse= zeros(length(dice_years))
           annual_marginal_damages = zeros(number_samples, length(annual_years))
           pc_consumption= zeros(number_samples, length(annual_years))
           
           #Index for values that fail
           error_indices = []

           #Caluclate Social Cost of Methane for each parameter sample from MCMC.
           for i in 1:number_samples

            try

               #FUND crashes with negative tempreature values. A couple chains give one or two years with temperatures that are -0.018.  Set these to 0.0.
               # Also set thees negative temperature values to 0.0 for DICE to maintina consistency across models.
               if minimum(base_temp[i, :]) < 0.0 || minimum(pulse_temp[i, :]) < 0.0
                   base_temp[i, find(base_temp[i, :] .< 0)] = 1e-15
                   pulse_temp[i, find(pulse_temp[i, :] .< 0)] = 1e-15
               end

               #Assign Sneasy Output to DICE parameters for Chain i
               setparameter(dice_base, :damages, :TATM, base_temp[i, :])
               setparameter(dice_pulse, :damages, :TATM, pulse_temp[i, :])

               #Run two versions of DICE
               run(dice_base)
               run(dice_pulse)

               #Calculate damages and convert from trillions to dollars
               damage_base[:] = dice_base[:damages, :DAMAGES] .* 10^12
               damage_pulse[:] = dice_pulse[:damages, :DAMAGES] .* 10^12
               #damage_base[:] = dice_base[:neteconomy, :C] .* 10^12
               #damage_pulse[:] = dice_pulse[:neteconomy, :C] .* 10^12

               # Take out growth effect of run 2 by transforming the damage from run 2 into % of GDP of run 2, and then
               # multiplying that with GDP of run 1
               #damage_marginal = dice_marginal[:damages,:DAMFRAC] .*dice_base[:grosseconomy,:YGROSS] * 10^12

               ##Convert from megatonnes to tonne -NOTE FINAL RUNS USE 1 TONNE.
               #annual_marginal_damages= dice_interpolate((damage_marginal .- damage_base), start_year, end_year) ./ 10^6
               annual_marginal_damages[i,:] = dice_interpolate((damage_pulse .- damage_base), 5) #./ 10^6
               pc_consumption[i,:] = dice_interpolate(dice_base[:neteconomy, :CPC],5)
          
              catch
                annual_marginal_damages[i,:] = -9999.99
                pc_consumption[i,:] = -9999.99
               #Print iteration number to track progress.
               #println("Completed marginal damage calculation ", i, " of ", number_samples, ".")
               push!(error_indices, i)
               println("Index screwed up:", i)

              end
           end
           return annual_marginal_damages, pc_consumption, error_indices
       end


function scch4_dice(annual_marginal_damages::Array{Float64,2}, pc_consumption::Array{Float64,2}, constant::Bool, discount_rate::Float64, elast::Float64, prtp::Float64)
    
    pulse_year = 2020
    # Calculate number of SC-CH4 values to calculate.
    n_samples = size(annual_marginal_damages,1)
    pulse_index_annual = findin(collect(2010:2300), 2020)[1]
    annual_years = collect(2010:2300)
    discounted_marginal_annual_damages = zeros(n_samples, length(annual_years))
    scch4_annual_share = zeros(n_samples, length(annual_years))
    cummulative_npv_damages = zeros(n_samples, length(annual_years))
    scch4 = zeros(n_samples)
    # Create array to hold discount factors
    df = zeros(length(annual_years))

    #Calculate a constant discount rate.
    if constant
        for t=pulse_index_annual:length(annual_years)
            #Set discount timestep for 5 year intervals
            tt = annual_years[t]
            x = 1 / ((1. + discount_rate)^(tt-pulse_year))
            df[t] = x
        end
    end

    # Loop through each marginal damage estimate.
    for i = 1:n_samples
        
        #Should a constant discount rate be used?
        if constant == false
            x = 1.0
            for t=pulse_index_annual:length(annual_years)
                df[t] = x
                gr = (pc_consumption[i,t] - pc_consumption[i,t-1]) / pc_consumption[i,t-1]
                x = x / (1. + prtp + elast * gr)
            end
        end

        discounted_marginal_annual_damages[i,:] = annual_marginal_damages[i,:] .* df

        scch4[i] = sum(discounted_marginal_annual_damages[i,:])

        # Caluclate cumulative NPV damages and share of SC-CH4 achieved at each timestep.
        cummulative_npv_damages[i,1] = discounted_marginal_annual_damages[i,1]
        scch4_annual_share[i,1] = cummulative_npv_damages[i,1] / scch4[i]
        
        for t = 2:length(annual_years)
            cummulative_npv_damages[i,t] = cummulative_npv_damages[i,t-1] + discounted_marginal_annual_damages[i,t]
            scch4_annual_share[i,t] = cummulative_npv_damages[i,t] / scch4[i]
        end

        #println("Finsihed run ", i, " of ", n_samples)
    end
    return scch4, discounted_marginal_annual_damages, cummulative_npv_damages, scch4_annual_share
end

#-----------------------------------------------------------------------
#Calculate Damages for Broken Correlations with DICE for RCP 8.5
#-----------------------------------------------------------------------
#SET UP DIRECTORY TO SAVE INTO
model_name = "sneasy_magicc"
mcmc_file_name = "5_mill_cauchy_final_July6"

model_name = "sneasy_hector" 
mcmc_file_name = "5_mill_cauchy_final_July6" #RF_uncert_NEWrf_hetero_Dec7_wider_init
#mcmc_file_name = "UNIFORM_ECS_5MILL"

model_name = "sneasy_fund"
mcmc_file_name = "5_mill_cauchy_final_July6" #RF_uncert_NEWrf_hetero_Dec9_wider_init

model_name = "sneasy_fair"
mcmc_file_name = "5_mill_cauchy_final_July6" #shorter version "NO_NATURAL_EMISS_RF_uncert_NEWrf"


rcp_scenario = "RCP85" #"RCP3PD"

model_scenario = "roe_baker"
model_scenario = "base_case"
model_scenario = "broken_correlations"
model_scenario = "old_RF"
model_scenario = "weitzman_damages"
#mkdir(output_directory)

output_directory = joinpath("scch4/dice/scch4_dice_results", model_name, rcp_scenario, mcmc_file_name, model_scenario)
mkpath(output_directory)

using DataFrames
using Interpolations
using RCall
using CSVFiles

#R"""
#library(ggplot2)
#library(reshape2)
#"""

Conf_lvl_1 = 0.95
Conf_lvl_2 = 0.98

include("helper_functions.jl")
include("src/dice_2013/src/dice2013_sneasyrun.jl")

# BASE CASE
#base_temp = convert(Array{Float64,2}, readtable("climate_ar1/base_case/sneasy_magicc/climate_ar1_results/4_5_mill_newRF_hetero_icecore_Oct4_RCP85/full_mcmc/base_temp.csv"));
#marginal_temp = convert(Array{Float64,2}, readtable("climate_ar1/base_case/sneasy_magicc/climate_ar1_results/4_5_mill_newRF_hetero_icecore_Oct4_RCP85/full_mcmc/marginal_temp.csv"));

# NO RF
base_temp = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/base_temp.csv"))));
#marginal_temp = convert(Array{Float64,2}, readtable(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/marginal_temp.csv")));
# Using CSVFiles
marginal_temp = convert(Array{Float64,2}, DataFrame(load(joinpath("climate_ar1",model_scenario,model_name,"climate_ar1_results",mcmc_file_name,rcp_scenario,"full_mcmc/marginal_temp.csv"))));
#TODO: FIX TRANPOSE ISSUE WITH confidence_int FUNCTION (WORKS, JUST ANNOYING)

years_scch4 = collect(2010:2300);
#This wants temp array as [mcmc run, year] so it would be [10,000 x 536]
marg_damages, pc_cons, err_index = scch4_damages_dice(base_temp, marginal_temp)

# Remove values that throw an error
correct_indices = collect(1:size(base_temp,1))
# Remove indices here that throw an error
deleteat!(correct_indices, err_index)
# Create new marg damages and pc_cons vals 
marg_damages = marg_damages[correct_indices,:]
pc_cons = pc_cons[correct_indices,:]

# Sometimes big neg value. find those indices
min_marg = zeros(size(marg_damages,1))
for i = 1:length(min_marg)
  min_marg[i] = minimum(marg_damages[i,:])
end

ham=find(x-> x >= -1.0, min_marg)

marg_damages = marg_damages[ham, :]
pc_cons = pc_cons[ham,:]

# 2.5% constant discount
scch4_25, annual_npv_damage_25, cumm_npv_damage_25, scch4_share_25 = scch4_dice(marg_damages, pc_cons, true, 0.025, 1.5, 0.015)
conf_cumm_dam_25 = confidence_int(years_scch4, cumm_npv_damage_25', Conf_lvl_1, Conf_lvl_2)
conf_share_25 = confidence_int(years_scch4, scch4_share_25', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_25 = confidence_int(years_scch4, annual_npv_damage_25', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_25.csv"), scch4_25)
writecsv(joinpath(output_directory, "annual_npv_damage_25.csv"), annual_npv_damage_25)
#writecsv(joinpath(output_directory, "cumm_npv_damage_25.csv"), cumm_npv_damage_25)
#writecsv(joinpath(output_directory, "scch4_share_25.csv"), scch4_share_25)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_25.csv"), conf_cumm_dam_25)
CSVFiles.save(joinpath(output_directory, "conf_share_25.csv"), conf_share_25)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_25.csv"), conf_annual_npv_dam_25)

# 3.0% constant discount
scch4_3, annual_npv_damage_3, cumm_npv_damage_3, scch4_share_3 = scch4_dice(marg_damages, pc_cons, true, 0.03, 1.5, 0.015)
conf_cumm_dam_3 = confidence_int(years_scch4, cumm_npv_damage_3', Conf_lvl_1, Conf_lvl_2)
conf_share_3 = confidence_int(years_scch4, scch4_share_3', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_3 = confidence_int(years_scch4, annual_npv_damage_3', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_3.csv"), scch4_3)
writecsv(joinpath(output_directory, "annual_npv_damage_3.csv"), annual_npv_damage_3)
#writecsv(joinpath(output_directory, "cumm_npv_damage_3.csv"), cumm_npv_damage_3)
#writecsv(joinpath(output_directory, "scch4_share_3.csv"), scch4_share_3)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_3.csv"), conf_cumm_dam_3)
CSVFiles.save(joinpath(output_directory, "conf_share_3.csv"), conf_share_3)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_3.csv"), conf_annual_npv_dam_3)

# 5.0% constant discount
scch4_5, annual_npv_damage_5, cumm_npv_damage_5, scch4_share_5 = scch4_dice(marg_damages, pc_cons, true, 0.05, 1.5, 0.015)
conf_cumm_dam_5 = confidence_int(years_scch4, cumm_npv_damage_5', Conf_lvl_1, Conf_lvl_2)
conf_share_5 = confidence_int(years_scch4, scch4_share_5', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_5 = confidence_int(years_scch4, annual_npv_damage_5', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_5.csv"), scch4_5)
writecsv(joinpath(output_directory, "annual_npv_damage_5.csv"), annual_npv_damage_5)
#writecsv(joinpath(output_directory, "cumm_npv_damage_5.csv"), cumm_npv_damage_5)
#writecsv(joinpath(output_directory, "scch4_share_5.csv"), scch4_share_5)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_5.csv"), conf_cumm_dam_5)
CSVFiles.save(joinpath(output_directory, "conf_share_5.csv"), conf_share_5)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_5.csv"), conf_annual_npv_dam_5)

# 7.0% constant discount
scch4_7, annual_npv_damage_7, cumm_npv_damage_7, scch4_share_7 = scch4_dice(marg_damages, pc_cons, true, 0.07, 1.5, 0.015)
conf_cumm_dam_7 = confidence_int(years_scch4, cumm_npv_damage_7', Conf_lvl_1, Conf_lvl_2)
conf_share_7 = confidence_int(years_scch4, scch4_share_7', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_7 = confidence_int(years_scch4, annual_npv_damage_7', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_7.csv"), scch4_7)
writecsv(joinpath(output_directory, "annual_npv_damage_7.csv"), annual_npv_damage_7)
#writecsv(joinpath(output_directory, "cumm_npv_damage_7.csv"), cumm_npv_damage_7)
#writecsv(joinpath(output_directory, "scch4_share_7.csv"), scch4_share_7)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_7.csv"), conf_cumm_dam_7)
CSVFiles.save(joinpath(output_directory, "conf_share_7.csv"), conf_share_7)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_7.csv"), conf_annual_npv_dam_7)

# Ramsey Nordhaus (eta = 1.5, prtp = 0.015)
scch4_nordhaus, annual_npv_damage_nordhaus, cumm_npv_damage_nordhaus, scch4_share_nordhaus = scch4_dice(marg_damages, pc_cons, false, 0.07, 1.5, 0.015)
conf_cumm_dam_nordhaus = confidence_int(years_scch4, cumm_npv_damage_nordhaus', Conf_lvl_1, Conf_lvl_2)
conf_share_nordhaus = confidence_int(years_scch4, scch4_share_nordhaus', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_nordhaus = confidence_int(years_scch4, annual_npv_damage_nordhaus', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_nordhaus.csv"), scch4_nordhaus)
writecsv(joinpath(output_directory, "annual_npv_damage_nordhaus.csv"), annual_npv_damage_nordhaus)
#writecsv(joinpath(output_directory, "cumm_npv_damage_nordhaus.csv"), cumm_npv_damage_nordhaus)
#writecsv(joinpath(output_directory, "scch4_share_nordhaus.csv"), scch4_share_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_nordhaus.csv"), conf_cumm_dam_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_share_nordhaus.csv"), conf_share_nordhaus)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_nordhaus.csv"), conf_annual_npv_dam_nordhaus)

# Ramsey Sterm (eta = 1.0, prtp = 0.001)
scch4_stern, annual_npv_damage_stern, cumm_npv_damage_stern, scch4_share_stern = scch4_dice(marg_damages, pc_cons, false, 0.07, 1.0, 0.001)
conf_cumm_dam_stern = confidence_int(years_scch4, cumm_npv_damage_stern', Conf_lvl_1, Conf_lvl_2)
conf_share_stern = confidence_int(years_scch4, scch4_share_stern', Conf_lvl_1, Conf_lvl_2)
conf_annual_npv_dam_stern = confidence_int(years_scch4, annual_npv_damage_stern', Conf_lvl_1, Conf_lvl_2)

writecsv(joinpath(output_directory, "scch4_stern.csv"), scch4_stern)
writecsv(joinpath(output_directory, "annual_npv_damage_stern.csv"), annual_npv_damage_stern)
writecsv(joinpath(output_directory, "cumm_npv_damage_stern.csv"), cumm_npv_damage_stern)
writecsv(joinpath(output_directory, "scch4_share_stern.csv"), scch4_share_stern)
CSVFiles.save(joinpath(output_directory, "conf_cumm_dam_stern.csv"), conf_cumm_dam_stern)
CSVFiles.save(joinpath(output_directory, "conf_share_stern.csv"), conf_share_stern)
CSVFiles.save(joinpath(output_directory, "conf_annual_npv_dam_stern.csv"), conf_annual_npv_dam_stern)


scch4_test, aaaannual_npv_damage_7, aaacumm_npv_damage_7, aaascch4_share_7 = scch4_dice(marg_damages, pc_cons, true, 0.01, 1.5, 0.015)

spaghetti_damages_double(scch4_share_stern, scch4_share_nordhaus, conf_share_stern, conf_share_nordhaus, "dodgerblue", "red", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1.05], "Year", "Share", "Test")


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











OLD STUFF

spaghetti_damages_double(scch4_share_3, scch4_share_25, conf_share_3, conf_share_25, "dodgerblue", "red", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1.05], "Year", "Share", "Test")

#Share of SC-CH4
ham2 = spaghetti_damages_quad(scch4_share_25, scch4_share_3, scch4_share_5, scch4_share_7, conf_share_25, conf_share_3, conf_share_5, conf_share_7,"darkorange", "red", "dodgerblue", "forestgreen", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1.05], "Year", "Share", "Test")
R"""
png(filename = "share.png", type="cairo", units="in", width=9, height=8, res=200)
print($ham2)
dev.off()
"""

ham5 = spaghetti_damages_five(scch4_share_stern, scch4_share_25, scch4_share_3, scch4_share_5, scch4_share_7, conf_share_stern, conf_share_25, conf_share_3, conf_share_5, conf_share_7, conf_share_nordhaus, "darkorchid3", "darkorange", "red", "dodgerblue", "forestgreen", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1.05], "Year", "Share", "Test")

#Cummulative damages
spaghetti_damages_quad(cumm_npv_damage_7, cumm_npv_damage_5, cumm_npv_damage_3, cumm_npv_damage_25, conf_cumm_dam_7, conf_cumm_dam_5, conf_cumm_dam_3, conf_cumm_dam_25,"darkorange", "red", "dodgerblue", "forestgreen", 0.05, 2010, 2300, 2300, 500, [2010,2200], [0,1000], "Year", "Share", "Test")

#npv annual damages
ham=spaghetti_damages_quad(annual_npv_damage_25, annual_npv_damage_3, annual_npv_damage_5, annual_npv_damage_7, conf_annual_npv_dam_25, conf_annual_npv_dam_3, conf_annual_npv_dam_5, conf_annual_npv_dam_7,"darkorange", "red", "dodgerblue", "forestgreen", 0.05, 2010, 2300, 2300, 400, [2010,2200], [0,45], "Year", "Share", "Test")
R"""
png(filename = "npv_damags.png", type="cairo", units="in", width=9, height=8, res=200)
print($ham)
dev.off()
"""
