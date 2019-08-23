using Distributions
using DataArrays
using CSV
using CSVFiles

#include("../../../src/sneasy_magicc/sneasy_magicc.jl")
include("calibration/calibration_helper_functions.jl")
include("src/sneasy_magicc/sneasy_magicc.jl")

mcmc_data ="5_mill_cauchy_final_July6"
chain_size = "thin_100k"

function climate_m(mcmc_data::String, chain_size::String; mean_chain::Bool=false, maxpost_chain::Bool=false, Conf_lvl_1::Float64=0.90, Conf_lvl_2::Float64=0.98)

    #Read in MCMC results and Emissions Data
    #if mean_chain
    #    chain = readtable(joinpath("calibration/sneasy_magicc/mcmc_results/", mcmc_data, "mean_params.csv"), separator =',', header=true);
    #elseif maxpost_chain
    #    chain = readtable(joinpath("calibration/sneasy_magicc/mcmc_results/", mcmc_data, "maxpost_params.csv"), separator =',', header=true);
    #else
        chain = DataFrame(load(joinpath("calibration/sneasy_magicc/mcmc_results/", mcmc_data, string(chain_size, ".csv"))))
    #end

    # Options are "RCP85" and "RCP3PD"
    #rcp_scenario = "RCP3PD"
    rcp_scenario = "RCP85"

    #RCP_Emissions = readtable("calibration/data/RCP85_EMISSIONS.csv");
    RCP_Emissions = readtable(joinpath("calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)

    #Caluclate number of MCMC parameter samples (each column is one vector of parameters)
    #number_samples = size(chain,2)
    number_samples = size(chain,1)

    #rcp_chain = thin_chain_100k;
    #number_samples = size(rcp_chain,2)

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
  
    #marginal_heat_interior    =   Array(Float64, number_years, number_samples)
    #marginal_oceanco2         =   Array(Float64, number_years, number_samples)
    #marginal_lifech4          =   Array(Float64, number_years, number_samples)

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
    sneasy_base = getsneasy_magicc(rcp_scenario= rcp_scenario, start_year=start_year, end_year=end_year);
    sneasy_marginal = getsneasy_magicc(rcp_scenario =rcp_scenario, start_year=start_year, end_year=end_year);

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
    setparameter(sneasy_marginal, :ch4cyclemagicc, :CH4_emissions, ch4_marginal)
    #####################################################################################

    #Run Sneasy for each MCMC parameter sample.

    for i in 1:number_samples

        #p=chain[:,i]
        #p = thin_chain_100k[:,i]
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
        TAUINIT      = chain[rand_indices[i,17], 17]
        σ_ch4inst    = chain[rand_indices[i,18], 18]
        ρ_ch4inst    = chain[rand_indices[i,19], 19]
        σ_ch4ice     = chain[rand_indices[i,20], 20]
        ρ_ch4ice     = chain[rand_indices[i,21], 21]
        CH4_0        = chain[rand_indices[i,22], 22]
        CH4_Nat      = chain[rand_indices[i,23], 23]
        Tsoil        = chain[rand_indices[i,24], 24]
        Tstrat       = chain[rand_indices[i,25], 25]
        N2O_0        = chain[rand_indices[i,26], 26]
        F2x_CO₂      = chain[rand_indices[i,27], 27]
        rf_scale_CH₄ = chain[rand_indices[i,28], 28]
        
        #Calculate CO₂ RF scaling based on F2x_CO₂ value.
        scale_CO₂ = co2_rf_scale(F2x_CO₂, CO20, N2O_0)

        setparameter(sneasy_base, :doeclim, :t2co, S)
        setparameter(sneasy_base, :doeclim, :kappa, κ)
        #setparameter(sneasy_base, :doeclim, :T0, T0)
        #setparameter(sneasy_base, :doeclim, :H0, H0)
        setparameter(sneasy_base, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_base, :ccm, :Q10, Q10)
        setparameter(sneasy_base, :ccm, :Beta, beta)
        setparameter(sneasy_base, :ccm, :Eta, eta)
        setparameter(sneasy_base, :ccm, :atmco20, CO20)
        setparameter(sneasy_base, :radiativeforcing, :alpha, α)
        setparameter(sneasy_base, :rfch4magicc, :CH4_0, CH4_0)
        setparameter(sneasy_base, :rfch4magicc, :N₂O_0, N2O_0)
        setparameter(sneasy_base, :rfch4magicc, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_base, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_base, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_base, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_base, :ch4cyclemagicc, :TAUINIT, TAUINIT)
        setparameter(sneasy_base, :ch4cyclemagicc, :CH4_0, CH4_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)
        setparameter(sneasy_base, :ch4cyclemagicc, :TAUSOIL, Tsoil)
        setparameter(sneasy_base, :ch4cyclemagicc, :TAUSTRAT, Tstrat)
        setparameter(sneasy_base, :ch4cyclemagicc, :CH4_natural, CH4_Nat)

        setparameter(sneasy_marginal, :doeclim, :t2co, S)
        setparameter(sneasy_marginal, :doeclim, :kappa, κ)
        #setparameter(sneasy_marginal, :doeclim, :T0, T0)
        #setparameter(sneasy_marginal, :doeclim, :H0, H0)
        setparameter(sneasy_marginal, :doeclim, :F2x_CO₂, F2x_CO₂)
        setparameter(sneasy_marginal, :ccm, :Q10, Q10)
        setparameter(sneasy_marginal, :ccm, :Beta, beta)
        setparameter(sneasy_marginal, :ccm, :Eta, eta)
        setparameter(sneasy_marginal, :ccm, :atmco20, CO20)
        setparameter(sneasy_marginal, :radiativeforcing, :alpha, α) 
        setparameter(sneasy_marginal, :rfch4magicc, :CH4_0, CH4_0)
        setparameter(sneasy_marginal, :rfch4magicc, :N₂O_0, N2O_0)
        setparameter(sneasy_marginal, :rfch4magicc, :scale_CH₄, rf_scale_CH₄)
        setparameter(sneasy_marginal, :rfco2, :CO₂_0, CO20)
        setparameter(sneasy_marginal, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 dat
        setparameter(sneasy_marginal, :rfco2, :scale_CO₂, scale_CO₂)
        setparameter(sneasy_marginal, :ch4cyclemagicc, :TAUINIT, TAUINIT)
        setparameter(sneasy_marginal, :ch4cyclemagicc, :CH4_0, CH4_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)
        setparameter(sneasy_marginal, :ch4cyclemagicc, :TAUSOIL, Tsoil)
        setparameter(sneasy_marginal, :ch4cyclemagicc, :TAUSTRAT, Tstrat)
        setparameter(sneasy_marginal, :ch4cyclemagicc, :CH4_natural, CH4_Nat)

        #Run climate models and collect results.
        try 
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
        base_ch4[i,:]                 = sneasy_base[:ch4cyclemagicc, :CH4] .+ ar1_hetero_ch4
        base_heat_interior[i,:]       = sneasy_base[:doeclim, :heat_interior] .+ ar1_hetero_oceanheat .+ H0
        base_oceanco2[i,:]            = sneasy_base[:ccm, :atm_oc_flux] .+ norm_oceanco2
        #base_lifech4[i,:]             = sneasy_base[:ch4cyclemagicc, :TAU_OH]

        marginal_temp[i,:]            = sneasy_marginal[:doeclim, :temp] .+ ar1_hetero_temperature
        marginal_co2[i,:]             = sneasy_marginal[:ccm, :atmco2] .+ ar1_hetero_co2
        marginal_ch4[i,:]             = sneasy_marginal[:ch4cyclemagicc, :CH4] .+ ar1_hetero_ch4
        #marginal_heat_interior[:,i]   = sneasy_marginal[:doeclim, :heat_interior] + H0 + ar1_heat
        #marginal_oceanco2[:,i]        = sneasy_marginal[:ccm, :atm_oc_flux] + norm_oceanco2
        #marginal_lifech4[:,i]         = sneasy_marginal[:ch4cyclemagicc, :TAU_OH]

        # Normalize temperatures to be relative to the 1850:1870 mean.
        base_temp[i,:]      = base_temp[i,:] .- mean(base_temp[i, index_1861:index_1880])
        marginal_temp[i,:]  = marginal_temp[i,:] .- mean(marginal_temp[i, index_1861:index_1880])

    catch
        base_temp[i,:]= -9999.99
        base_co2[i,:]= -9999.99
        base_ch4[i,:]= -9999.99
        base_heat_interior[i,:] = -9999.99
        base_oceanco2[i,:]= -9999.99
        #base_lifech4[i,:] = -9999.99
        marginal_temp[i,:]= -9999.99
        marginal_co2[i,:] = -9999.99
        marginal_ch4[i,:]  = -9999.99
    end

        #Print iteration number to track progress.
        println(i)#println("Completed model evaluation ", i, " of ", number_samples)
    end

# Check if any runs threw an error.
error_index = findin(base_temp[1,:], -9999.99)

save chain indices!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1



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

















    if !mean_chain && !maxpost_chain
        ##Caluclate confidence intervals for base run only using 'confidence_int' function from 'helper_functions.jl' file.
        #Gives back data in form Year, Mean across all samples for each year, Upper value (first CI),
        #Lower value (first CI), #Upper value (second CI), Lower value (second CI).

        conf_base_temp     = confidence_int(years, base_temp, Conf_lvl_1, Conf_lvl_2)
        conf_base_co2      = confidence_int(years, base_co2, Conf_lvl_1, Conf_lvl_2)
        conf_base_ch4      = confidence_int(years, base_ch4, Conf_lvl_1, Conf_lvl_2)
        conf_base_heatint  = confidence_int(years, base_heat_interior, Conf_lvl_1, Conf_lvl_2)
        conf_base_oceanco2 = confidence_int(years, base_oceanco2, Conf_lvl_1, Conf_lvl_2)
        conf_base_lifech4  = confidence_int(years, base_lifech4, Conf_lvl_1, Conf_lvl_2)

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
