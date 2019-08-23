using Distributions
using DataArrays
using CSV
using Distributions
using CSVFiles

include("src/sneasy_fund/sneasy_fund.jl")
include("calibration/calibration_helper_functions.jl")

mcmc_data ="5_mill_cauchy_final_July6"
chain_size = "mean_params"

function climate_m(mcmc_data::String, chain_size::String; mean_chain::Bool=false, maxpost_chain::Bool=false, Conf_lvl_1::Float64=0.90, Conf_lvl_2::Float64=0.98)

    #Read in MCMC results and Emissions Data
    #if mean_chain
        chain = DataFrame(load(joinpath("calibration/sneasy_fund/mcmc_results/", mcmc_data, string(chain_size, ".csv"))))
    #elseif maxpost_chain
        #chain = readtable(joinpath("calibration/sneasy_hector/mcmc_results/", mcmc_data, "maxpost_params.csv"), separator =',', header=true);
    #else
    #    chain = readtable(joinpath("calibration/sneasy_hector/mcmc_results/", mcmc_data, string(chain_size, ".csv")), separator =',', header=true);
    #end
    rcp_scenario = "RCP85"

     #RCP_Emissions = readtable("calibration/data/RCP85_EMISSIONS.csv");
    RCP_Emissions = readtable(joinpath("calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)
    #Caluclate number of MCMC parameter samples (each column is one vector of parameters)
    #number_samples = size(chain,1)

    number_samples = 100_000

    ###################################################################################
    #Number of years to run climate model
    start_year = 1765
    end_year = 2300
    number_years = length(start_year:end_year)

    #Create vector of years
    years = collect(start_year:end_year);

    # Get indices for 1850, 1870 to normalize temperature results.
    index_1861, index_1880 = findin(collect(start_year:end_year), [1861, 1880])

    # Load calibration data (for measurement errors).
    calibration_data = load_calibration_data(1850, 2017)

    #Marginal run adds one extra megatonne of CH4 that is scaled to a single tonne in FUND (to calculate the Social Cost of Methane).

    #Initialize arrays to store Sneasy results.
    base_temp               =   Array(Float64, number_samples, number_years);
    base_co2                =   Array(Float64, number_samples, number_years);
    base_ch4                =   Array(Float64, number_samples, number_years);
    base_heat_interior      =   Array(Float64, number_samples, number_years);
    base_oceanco2           =   Array(Float64, number_samples, number_years);
    #base_lifech4            =   Array(Float64, number_samples, number_years);

    marginal_temp           =   Array(Float64, number_samples, number_years);
    marginal_co2            =   Array(Float64, number_samples, number_years);
    marginal_ch4            =   Array(Float64, number_samples, number_years);
    #marginal_heat_interior  =   Array(Float64, number_years, number_samples);
    #marginal_oceanco2       =   Array(Float64, number_years, number_samples);
    #marginal_lifech4        =   Array(Float64, number_years, number_samples);

    ###################################################################################

    #####################################################################################

    # Allocate vectors to hold simulated noise.
    norm_oceanco2 = zeros(number_years);
    ar1_hetero_temperature= zeros(number_years);
    ar1_hetero_oceanheat= zeros(number_years);
    ar1_hetero_co2 = zeros(number_years);
    ar1_hetero_ch4 = zeros(number_years);    

    # Replicate errors for years without observations over model time horizon.
    obs_error_temperature = replicate_errors(start_year, end_year, calibration_data[:hadcrut_err]);
    obs_error_oceanheat = replicate_errors(start_year, end_year, calibration_data[:obs_ocheatsigma]);
    # NOTE: CO2 Has constant values for ice core and Mauna Loa (Mauna Loa starts in 1959)
    #obs_error_co2 = vcat(ones(length(start_year:1958)).*1.2, ones(length(1959:end_year)).*0.12);
    # NOTE CH4 ice core has constant error value
    obs_error_ch4_inst = replicate_errors(start_year, end_year, calibration_data[:ch4inst_sigma]);

    ###################################################################################
    #Get two versions of climate model (base run and marginal run with extra CH4).
    sneasy_base = getsneasy_fund(rcp_scenario= rcp_scenario, start_year=start_year, end_year=end_year);
    sneasy_marginal = getsneasy_fund(rcp_scenario =rcp_scenario, start_year=start_year, end_year=end_year);

    #Create Array of emissions data for base run.
    ch4_base = convert(Array, RCP_Emissions[:CH4]);

    #Set CH4 emissions in marginal model to have extra Megatonne of Ch4 emissions in 2020.
    ch4_marginal = convert(Array, copy(RCP_Emissions[:CH4]));
    marginal_year = 2020
    start_year_index, marginal_year_index, end_year_index = findin(RCP_Emissions[:YEARS], [start_year, marginal_year, end_year])

    # EMISSIONS IN UNITS MEGATONNES. ADD 1 TONNE.
    ch4_marginal[marginal_year_index] = ch4_marginal[marginal_year_index] + 1.0e-6

    #Extract emission values for the 1850:2300 period.
    ch4_marginal = ch4_marginal[start_year_index:end_year_index]

    #Set the CH4 emissions parameter to pick up the extra emissions in the marginal run.
    setparameter(sneasy_marginal, :ch4cycle, :globch4, ch4_marginal)

      κ = chain[2,1]
        α = chain[3,1]
        Q10 = chain[4,1]
        beta = chain[5,1]
        eta = chain[6,1]
        T0 = chain[7,1]
        H0 = chain[8,1]
        CO20 = chain[9,1]
        σ_temp = chain[10,1]
        σ_ocheat = chain[11,1]
        σ_co2inst = chain[12,1]
        σ_co2ice = chain[13,1]
        ρ_temp = chain[14,1]
        ρ_ocheat = chain[15,1]
        ρ_co2inst = chain[16,1]
        lifech4 = chain[17,1]
        σ_ch4inst = chain[18,1]
        ρ_ch4inst = chain[19,1]
        σ_ch4ice = chain[20,1]
        ρ_ch4ice = chain[21,1]
        CH4_0 = chain[22,1]
        N2O_0 = chain[23,1]
        F2x_CO₂ = chain[24,1]
        rf_scale_CH₄ = chain[25,1]
        
        #Calculate CO₂ RF scaling based on F2x_CO₂ value.
        scale_CO₂ = co2_rf_scale(F2x_CO₂, CO20, N2O_0)

    # Set all uncertain parameters except for equilibrium climate sensitivity to their mean values.
        setparameter(sneasy_base, :doeclim, :kappa, κ)
        setparameter(sneasy_base, :doeclim, :T0, T0)
        setparameter(sneasy_base, :doeclim, :H0, H0)
        setparameter(sneasy_base, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_base, :ccm, :Q10, Q10)
        setparameter(sneasy_base, :ccm, :Beta, beta)
        setparameter(sneasy_base, :ccm, :Eta, eta)
        setparameter(sneasy_base, :ccm, :atmco20, CO20)
        setparameter(sneasy_base, :radiativeforcing, :alpha, α)
        setparameter(sneasy_base, :rfch4, :CH₄_0, CH4_0)
        setparameter(sneasy_base, :rfch4, :N₂O_0, N2O_0)
        setparameter(sneasy_base, :rfch4, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_base, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_base, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_base, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_base, :ch4cycle, :lifech4, lifech4)
        setparameter(sneasy_base, :ch4cycle, :ch4pre, CH4_0)

        setparameter(sneasy_marginal, :doeclim, :kappa, κ)
        setparameter(sneasy_marginal, :doeclim, :T0, T0)
        setparameter(sneasy_marginal, :doeclim, :H0, H0)
        setparameter(sneasy_marginal, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_marginal, :ccm, :Q10, Q10)
        setparameter(sneasy_marginal, :ccm, :Beta, beta)
        setparameter(sneasy_marginal, :ccm, :Eta, eta)
        setparameter(sneasy_marginal, :ccm, :atmco20, CO20)
        setparameter(sneasy_marginal, :radiativeforcing, :alpha, α)
        setparameter(sneasy_marginal, :rfch4, :CH₄_0, CH4_0)
        setparameter(sneasy_marginal, :rfch4, :N₂O_0, N2O_0)
        setparameter(sneasy_marginal, :rfch4, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_marginal, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_marginal, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_marginal, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_marginal, :ch4cycle, :lifech4, lifech4)
        setparameter(sneasy_marginal, :ch4cycle, :ch4pre, CH4_0)

    #Construct sample climate sensitivity values from Roe and Baker (use EPA approximation).
     # This one anthoff told me (sent from marten?)
    #norm_dist2 = Truncated(Normal(0.62, 0.18), -0.2, 0.88)

    # This one from marten paper (Estimating thesocialcostofnon-CO2 GHG emissions: Methane and nitrous oxide)
    norm_dist = Truncated(Normal(0.6198, 0.1841), -0.2, 0.88)

    RB_ECS_sample = 1.2 ./ (1 .- rand(norm_dist, number_samples))
    #####################################################################################

    #Run Sneasy for each MCMC parameter sample.

    for i in 1:length(RB_ECS_sample) #number_samples

        S = RB_ECS_sample[i]

        #Set model parameters to values from current MCMC sample.
        setparameter(sneasy_base, :doeclim, :t2co, S)
        setparameter(sneasy_marginal, :doeclim, :t2co, S)

    try
        #Run climate models and collect results.
        run(sneasy_base)
        run(sneasy_marginal)

        ar1_hetero_temperature[:] = ar1_hetero_sim(number_years, ρ_temp, sqrt(obs_error_temperature.^2 .+ σ_temp^2))
        ar1_hetero_co2[:]         = co2_mixed_noise(start_year, end_year, σ_co2ice, σ_co2inst, 1.2, 0.12, ρ_co2inst)
        ar1_hetero_ch4[:]         = ch4_mixed_noise(start_year, end_year, ρ_ch4ice, σ_ch4ice, 15.0, ρ_ch4inst, σ_ch4inst, obs_error_ch4_inst)
        ar1_hetero_oceanheat[:]   = ar1_hetero_sim(number_years, ρ_ocheat, sqrt(obs_error_oceanheat.^2 .+ σ_ocheat^2))
        norm_oceanco2[:]          = rand(Normal(0,0.4*sqrt(10)), number_years)
        #else
        #    ar1_temp      = zeros(number_years)
        #    ar1_co2       = zeros(number_years)
        #    ar1_ch4       = zeros(number_years)
        #    ar1_heat      = zeros(number_years)
        #    norm_oceanco2 = zeros(number_years)
        #end

        base_temp[i,:]                = sneasy_base[:doeclim, :temp] .+ ar1_hetero_temperature .+ T0
        base_co2[i,:]                 = sneasy_base[:ccm, :atmco2] .+ ar1_hetero_co2
        base_ch4[i,:]                 = sneasy_base[:ch4cycle, :acch4] .+ ar1_hetero_ch4
        base_heat_interior[i,:]       = sneasy_base[:doeclim, :heat_interior] .+ ar1_hetero_oceanheat .+ H0
        base_oceanco2[i,:]            = sneasy_base[:ccm, :atm_oc_flux] .+ norm_oceanco2

        marginal_temp[i,:]            = sneasy_marginal[:doeclim, :temp] .+ ar1_hetero_temperature
        marginal_co2[i,:]             = sneasy_marginal[:ccm, :atmco2] .+ ar1_hetero_co2
        marginal_ch4[i,:]             = sneasy_marginal[:ch4cycle, :acch4] .+ ar1_hetero_ch4
        #marginal_heat_interior[:,i]   = sneasy_marginal[:doeclim, :heat_interior] + H0 + ar1_heat
        #marginal_oceanco2[:,i]        = sneasy_marginal[:ccm, :atm_oc_flux] + norm_oceanco2
    

        # Normalize temperatures to be relative to the 1850:1870 mean.
        base_temp[i,:]      = base_temp[i,:] .- mean(base_temp[i, index_1861:index_1880])
        marginal_temp[i,:]  = marginal_temp[i,:] .- mean(marginal_temp[i, index_1861:index_1880])

        #Print iteration number to track progress.
    catch
        println("Run ", i, " screwed up")
        base_temp[i,:]= -9999.99
        base_co2[i,:]= -9999.99
        base_ch4[i,:]= -9999.99
        base_heat_interior[i,:] = -9999.99
        base_oceanco2[i,:]= -9999.99
        marginal_temp[i,:]= -9999.99
        marginal_co2[i,:] = -9999.99
        marginal_ch4[i,:]  = -9999.99
    end
        #println("Completed climate model evaluation ", i, " of ", number_samples)
end

error_index = findin(base_temp[1,:], -9999.99)

Conf_lvl_1 = 0.95
Conf_lvl_2 = 0.98

    if !mean_chain && !maxpost_chain
        ##Caluclate confidence intervals for base run only using 'confidence_int' function from 'helper_functions.jl' file.
        #Gives back data in form Year, Mean across all samples for each year, Upper value (first CI),
        #Lower value (first CI), #Upper value (second CI), Lower value (second CI).

        conf_base_temp     = confidence_int(years, base_temp', Conf_lvl_1, Conf_lvl_2)
        conf_base_co2      = confidence_int(years, base_co2', Conf_lvl_1, Conf_lvl_2)
        conf_base_ch4      = confidence_int(years, base_ch4', Conf_lvl_1, Conf_lvl_2)
        conf_base_heatint  = confidence_int(years, base_heat_interior', Conf_lvl_1, Conf_lvl_2)
        conf_base_oceanco2 = confidence_int(years, base_oceanco2', Conf_lvl_1, Conf_lvl_2)
        #conf_base_lifech4  = confidence_int(years, base_lifech4, Conf_lvl_1, Conf_lvl_2)

        #Return base and marginal runs (with AR1 noise) and confidence interval calculations (for base run only).
        return years, base_temp, base_co2, base_ch4, base_heat_interior, base_oceanco2, base_lifech4, marginal_temp, marginal_co2, marginal_ch4, marginal_heat_interior, marginal_oceanco2, marginal_lifech4, conf_base_temp, conf_base_co2, conf_base_ch4, conf_base_heatint, conf_base_oceanco2, conf_base_lifech4

    else
        return years, base_temp, base_co2, base_ch4, base_heat_interior, base_oceanco2, base_lifech4, marginal_temp, marginal_co2, marginal_ch4, marginal_heat_interior, marginal_oceanco2, marginal_lifech4
    end

end
