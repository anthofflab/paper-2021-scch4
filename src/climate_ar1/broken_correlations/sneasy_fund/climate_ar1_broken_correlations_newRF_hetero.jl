using Distributions
using DataArrays
using CSV
using CSVFiles

include("calibration/calibration_helper_functions.jl")
include("src/sneasy_fund/sneasy_fund.jl")

mcmc_data ="5_mill_cauchy_final_July6"
chain_size = "thin_100k"

function climate_h(mcmc_data::String, chain_size::String; mean_chain::Bool=false, maxpost_chain::Bool=false, Conf_lvl_1::Float64=0.90, Conf_lvl_2::Float64=0.98)

    #Read in MCMC results and Emissions Data
    #if mean_chain
     #   chain = readtable(joinpath("calibration/sneasy_hector/mcmc_results/", mcmc_data, "mean_params.csv"), separator =',', header=true);
    #elseif maxpost_chain
    #    chain = readtable(joinpath("calibration/sneasy_hector/mcmc_results/", mcmc_data, "maxpost_params.csv"), separator =',', header=true);
    #else
        chain = DataFrame(load(joinpath("calibration/sneasy_fund/mcmc_results/", mcmc_data, string(chain_size, ".csv"))))
        #REMOVE THIS, COUPLE SAMPLES CAUSE SNEASY TO CRASH
        #chain = chain[:, [1:76090,76092:end]];
        #chain = chain[:, [1:82655,82657:82660,82662:end]];
    end

   # rcp_scenario = "RCP3PD"
    rcp_scenario = "RCP85"

    RCP_Emissions = readtable(joinpath("calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)

    #Caluclate number of MCMC parameter samples (each column is one vector of parameters)
    number_samples = size(chain,1)

    # Create random indices for each parameter to sample from (without repeating values).
    rand_indices = zeros(Int64, number_samples, size(chain)[2]);
    for i = 1:(size(chain)[2])
        rand_indices[:,i] = sample(1:number_samples, number_samples, replace=false)
    end

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

    # Create random indices for each parameter to sample from.
    #random_indices = rand(1:number_samples, 26, number_samples)

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

    #Get two versions of climate model (base run and marginal run with extra CH4).
    sneasy_base = getsneasy_fund(rcp_scenario= rcp_scenario, start_year=start_year, end_year=end_year);
    sneasy_marginal = getsneasy_fund(rcp_scenario =rcp_scenario, start_year=start_year, end_year=end_year);

    ###################################################################################
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

    # Counter in case non-physical results occur
    non_phys_count = 0
    #####################################################################################

    #Run Sneasy for each MCMC parameter sample.

    for i in 1:number_samples

        S            = chain[rand_indices[i,1], 1]
        κ            = chain[rand_indices[i,2], 2]
        α            = chain[rand_indices[i,3], 3]
        Q10          = chain[rand_indices[i,4], 4]
        beta         = chain[rand_indices[i,5], 5]
        eta          = chain[rand_indices[i,6], 6]
        T0           = chain[rand_indices[i,7], 7]
        H0           = chain[rand_indices[i,8], 8]
        CO20         = chain[rand_indices[i,9], 9]
        σ_temp       = chain[rand_indices[i,10], 10]
        σ_ocheat     = chain[rand_indices[i,11], 11]
        σ_co2inst    = chain[rand_indices[i,12], 12]
        σ_co2ice     = chain[rand_indices[i,13], 13]
        ρ_temp       = chain[rand_indices[i,14], 14]
        ρ_ocheat     = chain[rand_indices[i,15], 15]
        ρ_co2inst    = chain[rand_indices[i,16], 16]
        lifech4      = chain[rand_indices[i,17], 17]
        σ_ch4inst    = chain[rand_indices[i,18], 18]
        ρ_ch4inst    = chain[rand_indices[i,19], 19]
        σ_ch4ice     = chain[rand_indices[i,20], 20]
        ρ_ch4ice     = chain[rand_indices[i,21], 21]
        CH4_0        = chain[rand_indices[i,22], 22]
        N2O_0        = chain[rand_indices[i,23], 23]
        F2x_CO₂      = chain[rand_indices[i,24], 24]
        rf_scale_CH₄ = chain[rand_indices[i,25], 25]

        #Calculate CO₂ RF scaling based on F2x_CO₂ value.
        scale_CO₂ = co2_rf_scale(F2x_CO₂, CO20, N2O_0)

        #Set model parameters to values from current MCMC sample.
        setparameter(sneasy_base, :doeclim, :t2co, S)
        setparameter(sneasy_base, :doeclim, :kappa, κ)
        setparameter(sneasy_base, :doeclim, :F2x_CO₂,  F2x_CO₂)
        setparameter(sneasy_base, :ccm, :Q10, Q10)
        setparameter(sneasy_base, :ccm, :Beta, beta)
        setparameter(sneasy_base, :ccm, :Eta, eta)
        setparameter(sneasy_base, :ccm, :atmco20, CO20)
        setparameter(sneasy_base, :radiativeforcing, :alpha, α)
        setparameter(sneasy_base, :rfch4, :CH₄_0, CH4_0)
        setparameter(sneasy_base, :rfch4, :N₂O_0, N2O_0)
        setparameter(sneasy_base, :rfch4, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_base, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_base, :rfco2, :N₂O_0, N2O_0)
        setparameter(sneasy_base, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_base, :ch4cycle, :lifech4, lifech4)
        setparameter(sneasy_base, :ch4cycle, :ch4pre, CH4_0)

        setparameter(sneasy_marginal, :doeclim, :t2co, S)
        setparameter(sneasy_marginal, :doeclim, :kappa, κ)
        setparameter(sneasy_marginal, :doeclim, :F2x_CO₂,  F2x_CO₂)
        setparameter(sneasy_marginal, :ccm, :Q10, Q10)
        setparameter(sneasy_marginal, :ccm, :Beta, beta)
        setparameter(sneasy_marginal, :ccm, :Eta, eta)
        setparameter(sneasy_marginal, :ccm, :atmco20, CO20)
        setparameter(sneasy_marginal, :radiativeforcing, :alpha, α)
        setparameter(sneasy_marginal, :rfch4, :CH₄_0, CH4_0)
        setparameter(sneasy_marginal, :rfch4, :N₂O_0, N2O_0)
        setparameter(sneasy_marginal, :rfch4, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_marginal, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_marginal, :rfco2, :N₂O_0, N2O_0)
        setparameter(sneasy_marginal, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_marginal, :ch4cycle, :lifech4, lifech4)
        setparameter(sneasy_marginal, :ch4cycle, :ch4pre, CH4_0)

        try

            #Run climate models and collect results.
            run(sneasy_base)
            run(sneasy_marginal)

            #Set vectors of noise (impose noise for full MCMC chain)
            #if !mean_chain && !maxpost_chain
            ar1_hetero_temperature[:] = ar1_hetero_sim(number_years, ρ_temp, sqrt(obs_error_temperature.^2 .+ σ_temp^2))
            ar1_hetero_co2[:]         = co2_mixed_noise(start_year, end_year, σ_co2ice, σ_co2inst, 1.2, 0.12, ρ_co2inst)

            ar1_hetero_ch4[:]         = ch4_mixed_noise(start_year, end_year, ρ_ch4ice, σ_ch4ice, 15.0, ρ_ch4inst, σ_ch4inst, obs_error_ch4_inst)

            ar1_hetero_oceanheat[:]   = ar1_hetero_sim(number_years, ρ_ocheat, sqrt(obs_error_oceanheat.^2 .+ σ_ocheat^2))
            norm_oceanco2[:]          = rand(Normal(0,0.4*sqrt(10)), number_years)

            base_temp[i,:]                = sneasy_base[:doeclim, :temp] .+ ar1_hetero_temperature .+ T0
            base_co2[i,:]                 = sneasy_base[:ccm, :atmco2] .+ ar1_hetero_co2
            base_ch4[i,:]                 = sneasy_base[:ch4cycle, :acch4] .+ ar1_hetero_ch4
            base_heat_interior[i,:]       = sneasy_base[:doeclim, :heat_interior] .+ ar1_hetero_oceanheat .+ H0
            base_oceanco2[i,:]            = sneasy_base[:ccm, :atm_oc_flux] .+ norm_oceanco2

            marginal_temp[i,:]            = sneasy_marginal[:doeclim, :temp] .+ ar1_hetero_temperature
            marginal_co2[i,:]             = sneasy_marginal[:ccm, :atmco2] .+ ar1_hetero_co2
            marginal_ch4[i,:]             = sneasy_marginal[:ch4cycle, :acch4] .+ ar1_hetero_ch4

            # Normalize temperatures to be relative to the 1850:1870 mean.
            base_temp[i,:]      = base_temp[i,:] .- mean(base_temp[i, index_1861:index_1880])
            marginal_temp[i,:]  = marginal_temp[i,:] .- mean(marginal_temp[i, index_1861:index_1880])

        catch

            println("Model evaluation ", i, " of ", number_samples, " produced non-physical results.")
            base_temp[i,:] = -9999.99
            base_co2[i,:] = -9999.99
            base_ch4[i,:] = -9999.99
            base_heat_interior[i,:] = -9999.99
            base_oceanco2[i,:] = -9999.99

            marginal_temp[i,:] = -9999.99
            marginal_co2[i,:] = -9999.99
            marginal_ch4[i,:] = -9999.99
            #marginal_heat_interior[:,i] = -9999.99
            #marginal_oceanco2[:,i] = -9999.99
            #marginal_lifech4[:,i] = -9999.99

            # Add to count for non-physical model runs.
            non_phys_count += 1
        end
        #println("Completed model evaluation ", i, " of ", number_samples)
    end

    # Filter out non-physical model runs set to -9999.99.
    result_indices = find(x-> x != -9999.99, base_co2[1,:])

    base_temp           = base_temp[:, result_indices];
    base_co2            = base_co2[:, result_indices];
    base_ch4            = base_ch4[:, result_indices];
    base_heat_interior  = base_heat_interior[:, result_indices];
    base_oceanco2       = base_oceanco2[:, result_indices];
    base_lifech4        = base_lifech4[:, result_indices];

    marginal_temp           = marginal_temp[:, result_indices];
    marginal_co2            = marginal_co2[:, result_indices];
    marginal_ch4            = marginal_ch4[:, result_indices];
    #marginal_heat_interior  = marginal_heat_interior[:, result_indices];
    #marginal_oceanco2       = marginal_oceanco2[:, result_indices];
    #marginal_lifech4        = marginal_lifech4[:, result_indices];


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# SOME PARAM COMBINATIONS GIVE STRONGLY NEGATIVE TEMPS
# REMOVE ANYTHING WITH TEMP < 2 °C.
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------


min_temp = zeros(100_000)
for t = 1:100_000
    min_temp[t] = minimum(base_temp[t,:])
end

good_indices = find(x -> x > -2.0, min_temp)
bad_indices = find(x -> x < -2.0, min_temp)

base_temp           = base_temp[good_indices, :];
base_co2            = base_co2[good_indices, :];
base_ch4            = base_ch4[good_indices, :];
base_heat_interior  = base_heat_interior[good_indices, :];
base_oceanco2       = base_oceanco2[good_indices, :];
base_lifech4        = base_lifech4[good_indices, :];

marginal_temp           = marginal_temp[good_indices, :];
marginal_co2            = marginal_co2[good_indices, :];
marginal_ch4            = marginal_ch4[good_indices, :];


Conf_lvl_1 = 0.95
Conf_lvl_2 = 0.98

        conf_base_temp     = confidence_int(years, base_temp', Conf_lvl_1, Conf_lvl_2)
        conf_base_co2      = confidence_int(years, base_co2', Conf_lvl_1, Conf_lvl_2)
        conf_base_ch4      = confidence_int(years, base_ch4', Conf_lvl_1, Conf_lvl_2)
        conf_base_heatint  = confidence_int(years, base_heat_interior', Conf_lvl_1, Conf_lvl_2)
        conf_base_oceanco2 = confidence_int(years, base_oceanco2', Conf_lvl_1, Conf_lvl_2)

rcp_scenario = "RCP85"#"RCP85"#
output_folder = joinpath("climate_ar1/broken_correlations/sneasy_fund/climate_ar1_results", "RF_uncert_NEWrf_hetero_Dec9_wider_init", rcp_scenario)

# Create directories to save results.
mkpath(output_folder)
mkdir(joinpath(output_folder, "full_mcmc"))


        #---------------------------------------------------------------------------
        # Save climate results using full MCMC chain with AR(1) noise superimposed.
        #---------------------------------------------------------------------------
        CSVFiles.save(joinpath(output_folder,"full_mcmc","years.csv"), header=true, DataFrame(years=years))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","base_temp.csv"), header=true, DataFrame(transpose(base_temp)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","base_co2.csv"), header=true, DataFrame(transpose(base_co2)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","base_ch4.csv"), header=true, DataFrame(transpose(base_ch4)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","base_heat_interior.csv"), header=true, DataFrame(transpose(base_heat_interior)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","base_oceanco2.csv"), header=true, DataFrame(transpose(base_oceanco2)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","marginal_temp.csv"), header=true, DataFrame(transpose(marginal_temp)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","marginal_co2.csv"), header=true, DataFrame(transpose(marginal_co2)))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","marginal_ch4.csv"), header=true, DataFrame(transpose(marginal_ch4)))
        #writetable(joinpath(output_folder,"full_mcmc","marginal_heat_interior.csv"), header=true, DataFrame(marginal_heat_interior))
        #writetable(joinpath(output_folder,"full_mcmc","marginal_oceanco2.csv"), header=true, DataFrame(marginal_oceanco2))
       # writetable(joinpath(output_folder,"full_mcmc","marginal_lifech4.csv"), header=true, DataFrame(marginal_lifech4))
        #Save results for confidence intervals.
        CSVFiles.save(joinpath(output_folder,"full_mcmc","conf_base_temp.csv"), header=true, DataFrame(conf_base_temp))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","conf_base_co2.csv"), header=true, DataFrame(conf_base_co2))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","conf_base_ch4.csv"), header=true, DataFrame(conf_base_ch4))
        CSVFiles.save(joinpath(output_folder,"full_mcmc","conf_base_heatint.csv"), header=true, DataFrame(conf_base_heatint))
        #Remove last year (2300) because values are NA
        CSVFiles.save(joinpath(output_folder,"full_mcmc","conf_base_oceanco2.csv"), header=true, DataFrame(conf_base_oceanco2[1:end-1,:]))










R"""
library(ggplot2)
library(reshape2)
"""

temp_plot = spaghetti(base_temp, conf_base_temp, calibration_data, "hadcrut_temperature", "red", 0.1, 1765, 2300, 400, [1850,2017], [-0.5,1.5], "Year", "Temp", "Ham")
ocheat_plot = spaghetti(base_heat_interior, conf_base_heatint, calibration_data, "obs_ocheat", "red", 0.1, 1765, 2300, 400, [1850,2017], [-100,100], "Year", "Temp", "Ham")
ch4_plot = spaghetti(base_ch4, conf_base_ch4, calibration_data, "obs_ch4inst", "dodgerblue", 0.2, 1765, 2300, 900, [1850,2017], [500,2000], "Year", "Temp", "Ham")
R"""
$ch4_plot = $ch4_plot + geom_point(data=$calibration_data, aes_string(x="year", y="obs_ch4ice"), shape=21, size=2.5, colour="black", fill="peachpuff", alpha=1)
"""

co2_plot = spaghetti(base_co2, conf_base_co2, calibration_data, "obs_co2inst", "dodgerblue", 0.1, 1765, 2300, 400, [1850,2017], [250,400], "Year", "Temp", "Ham")
R"""
$co2_plot = $co2_plot + geom_point(data=$calibration_data, aes_string(x="year", y="obs_co2ice"), shape=21, size=2.5, colour="black", fill="peachpuff", alpha=1)
"""




function spaghetti(data, conf_data, observations, obs_variable, color, alpha, start_year, end_year, num_sims, x_range, y_range, x_title, y_title, main_title)

    R"""
    start_year = $start_year
    end_year = $end_year
    data = $data
    conf_data = $conf_data
    observations = $observations
    obs_variable = $obs_variable
    color= $color
    alpha = $alpha
    num_sims = $num_sims
    x_range = c($x_range[1], $x_range[2])
    y_range = c($y_range[1], $y_range[2])
    x_title = $x_title
    y_title = $y_title
    main_title = $main_title

    start_index = which(1765:2325==start_year)
    end_index = which(1765:2325==end_year)

    #Select 1000 values if num_sims is > 1000 (otherwise line overlap causes plot to become a solid color).

    rand_indices = sample(1:ncol(data), num_sims)
    reshaped_data = melt(data[start_index:end_index,rand_indices])

    plot_data = data.frame(Period = rep(start_year:end_year, times=num_sims), reshaped_data)
    colnames(plot_data) = c("Year", "NA", "Model_Run", "Value")
    plot_data[1:5,]
    
    #mean_data = data.frame(Period = ($start_year:$end_year), mean = $mean_data)
    p = ggplot()
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value", group="Model_Run"), colour=color, alpha=alpha)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="Mean_Chain"), colour="black", size=1.1)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="LowerConf_0.95"), colour="black", linetype="dashed", size=0.7)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="UpperConf_0.95"), colour="black", linetype="dashed", size=0.7)
    p = p + geom_point(data=observations, aes_string(x="year", y=obs_variable), shape=21, size=2.5, colour="black", fill="yellow", alpha=1)

    #p = p + xlim(c(2010,2305))
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