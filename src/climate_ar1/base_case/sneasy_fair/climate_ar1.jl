using Distributions
using DataArrays
using CSV
using CSVFiles

#include("../../../src/sneasy_magicc/sneasy_magicc.jl")
include("src/sneasy_fair/sneasy_fair_no_natural.jl")
#include("src/sneasy_fair/sneasy_fair_oldRF.jl")
include("helper_functions.jl")
include("calibration/calibration_helper_functions.jl")

mcmc_data ="5_mill_cauchy_final_July6"
chain_size = "thin_100k"

function climate_m(mcmc_data::String, chain_size::String; mean_chain::Bool=false, maxpost_chain::Bool=false, Conf_lvl_1::Float64=0.90, Conf_lvl_2::Float64=0.98)

    #Read in MCMC results and Emissions Data
    #if mean_chain
    #    chain = readtable(joinpath("calibration/sneasy_fair/mcmc_results/", mcmc_data, "mean_params.csv"), separator =',', header=true);
    #elseif maxpost_chain
    #    chain = readtable(joinpath("calibration/sneasy_fair/mcmc_results/", mcmc_data, "maxpost_params.csv"), separator =',', header=true);
    #else
        chain = DataFrame(load(joinpath("calibration/sneasy_fair/mcmc_results/", mcmc_data, string(chain_size, ".csv"))))
    #end

    # Options are "RCP85" and "RCP3PD"
    rcp_scenario = "RCP3PD"
    rcp_scenario = "RCP85"

    #RCP_Emissions = readtable("calibration/data/RCP85_EMISSIONS.csv");
    RCP_Emissions = readtable(joinpath("calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)

    #Caluclate number of MCMC parameter samples (each column is one vector of parameters)
    #number_samples = size(chain,2)
    number_samples = size(chain,1)

    #rcp_chain = thin_chain_10k;
    #number_samples = size(rcp_chain,2)
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
    sneasy_base = getsneasy_fair(rcp_scenario= rcp_scenario, start_year=start_year, end_year=end_year);
    sneasy_marginal = getsneasy_fair(rcp_scenario =rcp_scenario, start_year=start_year, end_year=end_year);

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
    setparameter(sneasy_marginal, :ch4cycle_fair, :fossil_emiss, ch4_marginal)
    #####################################################################################

    #Run Sneasy for each MCMC parameter sample.

    for i in 1:number_samples

        p=Array(chain[i,:])

        S = p[1]
        κ = p[2]
        α = p[3]
        Q10 = p[4]
        beta = p[5]
        eta = p[6]
        T0 = p[7]
        H0 = p[8]
        CO20 = p[9]
        σ_temp = p[10]
        σ_ocheat = p[11]
        σ_co2inst = p[12]
        σ_co2ice = p[13]
        ρ_temp = p[14]
        ρ_ocheat = p[15]
        ρ_co2inst = p[16]
        τ = p[17]
        σ_ch4inst = p[18]
        ρ_ch4inst = p[19]
        σ_ch4ice = p[20]
        ρ_ch4ice = p[21]    
        CH4_0 = p[22]
        CH4_Nat = p[23]
        N2O_0 = p[24]
        F2x_CO₂ = p[25]
        rf_scale_CH₄ = p[26]
        
        #Calculate CO₂ RF scaling based on F2x_CO₂ value.
        scale_CO₂ = co2_rf_scale(F2x_CO₂, CO20, N2O_0)

        setparameter(sneasy_base, :doeclim, :t2co, S)
        setparameter(sneasy_base, :doeclim, :kappa, κ)
        setparameter(sneasy_base, :doeclim, :T0, T0)
        setparameter(sneasy_base, :doeclim, :H0, H0)
        setparameter(sneasy_base, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_base, :ccm, :Q10, Q10)
        setparameter(sneasy_base, :ccm, :Beta, beta)
        setparameter(sneasy_base, :ccm, :Eta, eta)
        setparameter(sneasy_base, :ccm, :atmco20, CO20)
        setparameter(sneasy_base, :radiativeforcing, :alpha, α)
        setparameter(sneasy_base, :rfch4fair, :CH₄_0, CH4_0)
        setparameter(sneasy_base, :rfch4fair, :N₂O_0, N2O_0)
        setparameter(sneasy_base, :rfch4fair, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_base, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_base, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_base, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_base, :ch4cycle_fair, :τ, τ)
        setparameter(sneasy_base, :ch4cycle_fair, :CH₄_0, CH4_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)
        setparameter(sneasy_base, :ch4cycle_fair, :constant_natural_CH₄, CH4_Nat)
        setparameter(sneasy_base, :fair_trop_o3, :CH₄_0, CH4_0)

        setparameter(sneasy_marginal, :doeclim, :t2co, S)
        setparameter(sneasy_marginal, :doeclim, :kappa, κ)
        setparameter(sneasy_marginal, :doeclim, :T0, T0)
        setparameter(sneasy_marginal, :doeclim, :H0, H0)
        setparameter(sneasy_marginal, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_marginal, :ccm, :Q10, Q10)
        setparameter(sneasy_marginal, :ccm, :Beta, beta)
        setparameter(sneasy_marginal, :ccm, :Eta, eta)
        setparameter(sneasy_marginal, :ccm, :atmco20, CO20)
        setparameter(sneasy_marginal, :radiativeforcing, :alpha, α)
        setparameter(sneasy_marginal, :rfch4fair, :CH₄_0, CH4_0)
        setparameter(sneasy_marginal, :rfch4fair, :N₂O_0, N2O_0)
        setparameter(sneasy_marginal, :rfch4fair, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_marginal, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_marginal, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_marginal, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_marginal, :ch4cycle_fair, :τ, τ)
        setparameter(sneasy_marginal, :ch4cycle_fair, :CH₄_0, CH4_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)
        setparameter(sneasy_marginal, :ch4cycle_fair, :constant_natural_CH₄, CH4_Nat)
        setparameter(sneasy_marginal, :fair_trop_o3, :CH₄_0, CH4_0)
        
        run(sneasy_base)
        run(sneasy_marginal)

        # Create noise to superimpose on results.
        ar1_hetero_temperature[:] = ar1_hetero_sim(number_years, ρ_temp, sqrt(obs_error_temperature.^2 .+ σ_temp^2))
        ar1_hetero_co2[:]         = co2_mixed_noise(start_year, end_year, σ_co2ice, σ_co2inst, 1.2, 0.12, ρ_co2inst)

        ar1_hetero_ch4[:]         = ch4_mixed_noise(start_year, end_year, ρ_ch4ice, σ_ch4ice, 15.0, ρ_ch4inst, σ_ch4inst, obs_error_ch4_inst)

        ar1_hetero_oceanheat[:]   = ar1_hetero_sim(number_years, ρ_ocheat, sqrt(obs_error_oceanheat.^2 .+ σ_ocheat^2))
        norm_oceanco2[:]          = rand(Normal(0,0.4*sqrt(10)), number_years)

        base_temp[i,:]                = sneasy_base[:doeclim, :temp] .+ ar1_hetero_temperature .+ T0
        base_co2[i,:]                 = sneasy_base[:ccm, :atmco2] .+ ar1_hetero_co2
        base_ch4[i,:]                 = sneasy_base[:ch4cycle_fair, :CH₄] .+ ar1_hetero_ch4
        base_heat_interior[i,:]       = sneasy_base[:doeclim, :heat_interior] .+ ar1_hetero_oceanheat .+ H0
        base_oceanco2[i,:]            = sneasy_base[:ccm, :atm_oc_flux] .+ norm_oceanco2

        marginal_temp[i,:]            = sneasy_marginal[:doeclim, :temp] .+ ar1_hetero_temperature
        marginal_co2[i,:]             = sneasy_marginal[:ccm, :atmco2] .+ ar1_hetero_co2
        marginal_ch4[i,:]             = sneasy_marginal[:ch4cycle_fair, :CH₄] .+ ar1_hetero_ch4
        #marginal_heat_interior[:,i]   = sneasy_marginal[:doeclim, :heat_interior] + H0 + ar1_heat
        #marginal_oceanco2[:,i]        = sneasy_marginal[:ccm, :atm_oc_flux] + norm_oceanco2
        #marginal_lifech4[:,i]         = sneasy_marginal[:ch4cyclemagicc, :TAU_OH]

        # Normalize temperatures to be relative to the 1850:1870 mean.
        base_temp[i,:]      = base_temp[i,:] .- mean(base_temp[i, index_1861:index_1880])
        marginal_temp[i,:]  = marginal_temp[i,:] .- mean(marginal_temp[i, index_1861:index_1880])

   #= catch
        base_temp[:,i]= -9999.99
        base_co2[:,i]= -9999.99
        base_ch4[:,i]= -9999.99
        base_heat_interior[:,i] = -9999.99
        base_oceanco2[:,i]= -9999.99
        base_lifech4[:,i] = -9999.99
        marginal_temp[:,i]= -9999.99
        marginal_co2[:,i] = -9999.99
        marginal_ch4[:,i]  = -9999.99
    end
=#
        println(i)
        #Print iteration number to track progress.
        #println("Completed model evaluation ", i, " of ", number_samples)
    end

Conf_lvl_1 = 0.95
Conf_lvl_2 = 0.98

# Check if any runs threw an error.
error_index = findin(base_temp[1,:], -9999.99)


    if !mean_chain && !maxpost_chain
        ##Caluclate confidence intervals for base run only using 'confidence_int' function from 'helper_functions.jl' file.
        #Gives back data in form Year, Mean across all samples for each year, Upper value (first CI),
        #Lower value (first CI), #Upper value (second CI), Lower value (second CI).

        conf_base_temp     = confidence_int(years, base_temp', Conf_lvl_1, Conf_lvl_2)
        conf_base_co2      = confidence_int(years, base_co2', Conf_lvl_1, Conf_lvl_2)
        conf_base_ch4      = confidence_int(years, base_ch4', Conf_lvl_1, Conf_lvl_2)
        conf_base_heatint  = confidence_int(years, base_heat_interior', Conf_lvl_1, Conf_lvl_2)
        conf_base_oceanco2 = confidence_int(years, base_oceanco2', Conf_lvl_1, Conf_lvl_2)

        #Return base and marginal runs (with AR1 noise) and confidence interval calculations (for base run only).
        return years, base_temp, base_co2, base_ch4, base_heat_interior, base_oceanco2, base_lifech4, marginal_temp, marginal_co2, marginal_ch4, marginal_heat_interior, marginal_oceanco2, marginal_lifech4, conf_base_temp, conf_base_co2, conf_base_ch4, conf_base_heatint, conf_base_oceanco2, conf_base_lifech4

    else
        return years, base_temp, base_co2, base_ch4, base_heat_interior, base_oceanco2, base_lifech4, marginal_temp, marginal_co2, marginal_ch4, marginal_heat_interior, marginal_oceanco2, marginal_lifech4
    end

end

#=
temp_hindcast = spaghetti(DataFrame(base_temp), conf_base_temp, calibration_data, "hadcrut_temperature", "red", 0.08, 1850, 2300, 600, [1850,2300], [-0.5, 10.5], "Year", "Degrees C", "Temperature Anomaly")
oceanheat_hindcast = spaghetti(DataFrame(base_heat_interior), conf_base_heatint, calibration_data, "obs_ocheat", "red", 0.08, 1850, 2020, 600, [1850,2020], [-150, 150], "Year", "10^22 J", "Ocean Heat")
co2_hindcast = spaghetti(DataFrame(base_co2), conf_base_co2, calibration_data, "obs_co2inst", "forestgreen", 0.08, 1850, 2020, 600, [1850,2020], [280, 420], "Year", "ppm", "Atmospheric CO2 Concentration")
R"""
$co2_hindcast = $co2_hindcast + geom_point(data=$calibration_data, aes_string(x="year", y="obs_co2ice"), shape=21, size=2.5, colour="black", fill="peachpuff", alpha=1)
"""
oceanco2_hindcast = spaghetti(DataFrame(base_oceanco2), conf_base_oceanco2, calibration_data, "obs_ocflux", "forestgreen", 0.08, 1850, 2020, 600, [1850,2020], [-10, 10], "Year", "GtC/yr", "Ocean Carbon Flux")
ch4_hindcast = spaghetti(DataFrame(base_ch4), conf_base_ch4, calibration_data, "obs_ch4inst", "darkorchid4", 0.08, 1850, 2020, 600, [1850,2020], [500, 2500], "Year", "MtCH4/yr", "Atmospheric CH4 Concentration")
R"""
$ch4_hindcast = $ch4_hindcast + geom_point(data=$calibration_data, aes_string(x="year", y="obs_ch4ice"), shape=21, size=2.5, colour="black", fill="peachpuff", alpha=1)
"""
=#
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
ch4_life_plot = spaghetti(base_lifech4, conf_base_lifech4, calibration_data, "obs_ch4inst", "dodgerblue", 0.1, 1765, 2300, 400, [1850,2017], [5,12], "Year", "Temp", "Ham")


ham_ocheat=zeros(536,10000); 
for t = 1:10000
    ham_ocheat[:,t] = base_heat_interior[:,t] .- mean(base_heat_interior[97:116,t])
end
conf_base_ham_heatint  = confidence_int(years, ham_ocheat, Conf_lvl_1, Conf_lvl_2)
ocheat_plot_ham = spaghetti(ham_ocheat, conf_base_ham_heatint, calibration_data, "obs_ocheat", "red", 0.1, 1765, 2300, 400, [1765,2017], [-100,100], "Year", "Temp", "Ham")


base_heat_interior .- mean(base_heat_interior[])

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