using CSVFiles
using DataFrames
using Distributions
using Mimi
using Statistics

# Load files with additional functions needed for analysis.
include(joinpath("..", "helper_functions.jl"))
include(joinpath("..", "..", "calibration", "calibration_helper_functions.jl"))

# Load the four versions of SNEASY-CH4.
include(joinpath("..", "create_models", "create_sneasy_fairch4.jl"))
include(joinpath("..", "create_models", "create_sneasy_fundch4.jl"))
include(joinpath("..", "create_models", "create_sneasy_hectorch4.jl"))
include(joinpath("..", "create_models", "create_sneasy_magiccch4.jl"))


function construct_sneasych4_ecs(climate_model::Symbol, rcp::String,  pulse_year::Int, pulse_size::Float64, end_year::Int)

    #---------------------------------------
    # Load and clean up some data.
    #---------------------------------------

    # Load RCP scenario emissions data.
    rcp_emissions = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp*"_emissions.csv"), skiplines_begin=36))

    # Crop emissions to proper time periods (1765-end_year)
    rcp_indices = findall((in)(collect(1765:end_year)), rcp_emissions.YEARS)
    rcp_emissions = rcp_emissions[rcp_indices, :]

    # Get years the model is run for (initializing it in 1765).
    model_years  = collect(1765:end_year)
    number_years = length(model_years)

    # Get indices for 1861-1880 to normalize temperature projections.
    indices_1861_1880 = findall((in)(collect(1861:1880)), model_years)

    # Load calibration data from 1850-2017 (measurement errors used in simulated noise).
    calibration_data = load_calibration_data(1850, 2017)

    # Pre-allocate vectors to hold simulated AR(1) + measurement error noise.
    norm_oceanco2   = zeros(number_years)
    ar1_temperature = zeros(number_years)
    ar1_oceanheat   = zeros(number_years)
    ar1_co2         = zeros(number_years)
    ar1_ch4         = zeros(number_years)

    # Replicate errors for years without observations over model time horizon (note, CO₂ and CH₄ ice-core have constant error estimates).
    obs_error_temperature = replicate_errors(1765, end_year, calibration_data.hadcrut_temperature_sigma)
    obs_error_oceanheat   = replicate_errors(1765, end_year, calibration_data.ocean_heat_sigma)
    obs_error_noaa_ch4    = replicate_errors(1765, end_year, calibration_data.noaa_ch4_sigma)

    # Set up marginal CH₄ emissions time series to have an extra emission pulse in a user-specified year (note: RCP CH₄ emissions in megatonnes).
    ch4_emissions_pulse = rcp_emissions.CH4
    pulse_year_index = findall(x -> x == pulse_year, rcp_emissions.YEARS)[1]
    ch4_emissions_pulse[pulse_year_index] = ch4_emissions_pulse[pulse_year_index] + pulse_size



    #-----------------------------------------------------------------------------------
    # Create two versions of SNEASY+CH4 based on the specific methane cycle selected.
    #-----------------------------------------------------------------------------------
    if climate_model == :sneasy_fair
        sneasych4_base  = create_sneasy_fairch4(rcp_scenario=rcp)
        sneasych4_pulse = create_sneasy_fairch4(rcp_scenario=rcp)
        update_param!(sneasych4_pulse, :fossil_emiss_CH₄, ch4_emissions_pulse)

    elseif climate_model == :sneasy_fund
        sneasych4_base  = create_sneasy_fundch4(rcp_scenario=rcp)
        sneasych4_pulse = create_sneasy_fundch4(rcp_scenario=rcp)
        update_param!(sneasych4_pulse, :globch4, ch4_emissions_pulse)

    elseif climate_model == :sneasy_hector
        sneasych4_base  = create_sneasy_hectorch4(rcp_scenario=rcp)
        sneasych4_pulse = create_sneasy_hectorch4(rcp_scenario=rcp)
        update_param!(sneasych4_pulse, :CH4_emissions, ch4_emissions_pulse)

    elseif climate_model == :sneasy_magicc
        sneasych4_base  = create_sneasy_magiccch4(rcp_scenario=rcp)
        sneasych4_pulse = create_sneasy_magiccch4(rcp_scenario=rcp)
        update_param!(sneasych4_pulse, :CH4_emissions, ch4_emissions_pulse)

    else
        error("Incorrect climate model selected. Options include :sneasy_fair, :sneasy_fund, :sneasy_hector, or :sneasy_magicc.")
    end


    #---------------------------------------------------------------------------------------------------------------------
    # Create functions to set parameters and return results that differ across the four methane cycle models.
    #---------------------------------------------------------------------------------------------------------------------

    # Create a function to set the uncertain methane cycle parameters.
    # Note: This function assumes parameters follow the same ordering as in the `create_log_posterior.jl` file.
    update_ch4_params! = create_update_ch4_function(climate_model)

    # Create a function to return projected CH₄ concentrations (component and variable names differ across models).
    get_ch4_results! = create_get_ch4_results_function(climate_model)


    #---------------------------------------------------------------------------------------------------------------------
    # Given user-specified settings, create a function to run SNEASY+CH4 over the equilibrium climate sensitivity samples.
    #---------------------------------------------------------------------------------------------------------------------

    function sneasych4_ecs(ecs_sample::Array{Float64,1}, mean_posterior_parameters::Array{Float64,1}, ci_interval_1::Float64, ci_interval_2::Float64)

        # Caluclate number of ECS parameter samples.
        number_samples = length(ecs_sample)

        # Pre-allocate arrays to store SNEASY+CH4 results.
        base_temperature  =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_co2          =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_ch4          =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_ocean_heat   =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_oceanco2     =   zeros(Union{Missing, Float64}, number_samples, number_years)

        pulse_temperature =   zeros(Union{Missing, Float64}, number_samples, number_years)
        pulse_co2         =   zeros(Union{Missing, Float64}, number_samples, number_years)

        # Assign names to individual mean posterior parameter sample (for convenience).
        σ_temperature     = mean_posterior_parameters[1]
        σ_ocean_heat      = mean_posterior_parameters[2]
        σ_CO₂inst         = mean_posterior_parameters[3]
        σ_CO₂ice          = mean_posterior_parameters[4]
        σ_CH₄inst         = mean_posterior_parameters[5]
        σ_CH₄ice          = mean_posterior_parameters[6]
        ρ_temperature     = mean_posterior_parameters[7]
        ρ_ocean_heat      = mean_posterior_parameters[8]
        ρ_CO₂inst         = mean_posterior_parameters[9]
        ρ_CH₄inst         = mean_posterior_parameters[10]
        ρ_CH₄ice          = mean_posterior_parameters[11]
        temperature_0     = mean_posterior_parameters[12]
        ocean_heat_0      = mean_posterior_parameters[13]
        CO₂_0             = mean_posterior_parameters[14]
        CH₄_0             = mean_posterior_parameters[15]
        N₂O_0             = mean_posterior_parameters[16]
        ECS               = mean_posterior_parameters[17]
        heat_diffusivity  = mean_posterior_parameters[18]
        rf_scale_aerosol  = mean_posterior_parameters[19]
        rf_scale_CH₄      = mean_posterior_parameters[20]
        F2x_CO₂           = mean_posterior_parameters[21]
        Q10               = mean_posterior_parameters[22]
        CO₂_fertilization = mean_posterior_parameters[23]
        CO₂_diffusivity   = mean_posterior_parameters[24]

        # Set all uncertain parameters in base model (except for ECS) to mean posterior value.
        update_param!(sneasych4_base, :kappa, heat_diffusivity)
        update_param!(sneasych4_base, :F2x_CO₂, F2x_CO₂)
        update_param!(sneasych4_base, :Q10, Q10)
        update_param!(sneasych4_base, :Beta, CO₂_fertilization)
        update_param!(sneasych4_base, :Eta, CO₂_diffusivity)
        update_param!(sneasych4_base, :atmco20, CO₂_0)
        update_param!(sneasych4_base, :CO₂_0, CO₂_0)
        update_param!(sneasych4_base, :N₂O_0, N₂O_0)
        update_param!(sneasych4_base, :rf_scale_CO₂, co2_rf_scale(F2x_CO₂, CO₂_0, N₂O_0))
        update_param!(sneasych4_base, :α, rf_scale_aerosol)
        update_ch4_params!(sneasych4_base, mean_posterior_parameters)

        # Set all uncertain parameters in pulse model (except for ECS) to mean posterior value.
        update_param!(sneasych4_pulse, :kappa, heat_diffusivity)
        update_param!(sneasych4_pulse, :F2x_CO₂, F2x_CO₂)
        update_param!(sneasych4_pulse, :Q10, Q10)
        update_param!(sneasych4_pulse, :Beta, CO₂_fertilization)
        update_param!(sneasych4_pulse, :Eta, CO₂_diffusivity)
        update_param!(sneasych4_pulse, :atmco20, CO₂_0)
        update_param!(sneasych4_pulse, :CO₂_0, CO₂_0)
        update_param!(sneasych4_pulse, :N₂O_0, N₂O_0)
        update_param!(sneasych4_pulse, :rf_scale_CO₂, co2_rf_scale(F2x_CO₂, CO₂_0, N₂O_0))
        update_param!(sneasych4_pulse, :α, rf_scale_aerosol)
        update_ch4_params!(sneasych4_pulse, mean_posterior_parameters)

        # For each sample ECS value, run the base and pulse versions of SNEASY+CH4 and store results.
        for i in 1:number_samples

            # Set ECS parameter for base and pulse versions of SNEASY+CH4.
            update_param!(sneasych4_base,  :t2co, ecs_sample[i])
            update_param!(sneasych4_pulse, :t2co, ecs_sample[i])

            # Wrap code in a try/catch statement in case extreme ECS value causes a model error.

            try
                # Run both models.
                run(sneasych4_base)
                run(sneasych4_pulse)

                # Create noise to superimpose on results using mean posterior statistical parameters and measurement noise (note: Both models use same estimated noise).
                ar1_temperature[:] = ar1_hetero_sim(number_years, ρ_temperature, sqrt.(obs_error_temperature.^2 .+ σ_temperature^2))
                ar1_co2[:]         = co2_mixed_noise(1765, end_year, σ_CO₂ice, σ_CO₂inst, 1.2, 0.12, ρ_CO₂inst)
                ar1_ch4[:]         = ch4_mixed_noise(1765, end_year, ρ_CH₄ice, σ_CH₄ice, 15.0, ρ_CH₄inst, σ_CH₄inst, obs_error_noaa_ch4)
                ar1_oceanheat[:]   = ar1_hetero_sim(number_years, ρ_ocean_heat, sqrt.(obs_error_oceanheat.^2 .+ σ_ocean_heat^2))
                norm_oceanco2[:]   = rand(Normal(0,0.4*sqrt(10)), number_years)

                # Store model projections resulting from parameter sample `i` for base model.
                base_temperature[i,:]  = sneasych4_base[:doeclim, :temp] .+ ar1_temperature .+ temperature_0
                base_co2[i,:]          = sneasych4_base[:ccm, :atmco2] .+ ar1_co2
                base_ocean_heat[i,:]   = sneasych4_base[:doeclim, :heat_interior] .+ ar1_oceanheat .+ ocean_heat_0
                base_oceanco2[i,:]     = sneasych4_base[:ccm, :atm_oc_flux] .+ norm_oceanco2
                base_ch4[i,:]          = get_ch4_results!(sneasych4_base) .+ ar1_ch4

                # Store tempeature and CO₂ projections resulting from parameter sample `i` for pulse model (used for estimating the SC-CH₄).
                pulse_temperature[i,:] = sneasych4_pulse[:doeclim, :temp] .+ ar1_temperature
                pulse_co2[i,:]         = sneasych4_pulse[:ccm, :atmco2] .+ ar1_co2

                # Normalize temperatures to be relative to the 1861-1880 mean.
                base_temperature[i,:]  = base_temperature[i,:] .- mean(base_temperature[i, indices_1861_1880])
                pulse_temperature[i,:] = pulse_temperature[i,:] .- mean(pulse_temperature[i, indices_1861_1880])

            catch

                # Set values to -99999.99 if model throws an error.
                base_temperature[i,:]  .= -99999.99
                base_co2[i,:]          .= -99999.99
                base_ocean_heat[i,:]   .= -99999.99
                base_oceanco2[i,:]     .= -99999.99
                base_ch4[i,:]          .= -99999.99
                pulse_temperature[i,:] .= -99999.99
                pulse_co2[i,:]         .= -99999.99
            end
        end

        # Distinguish between model indices that caused a model error or not.
        error_indices = findall(x-> x == -99999.99, base_temperature[:,1])
        good_indices  = findall(x-> x != -99999.99, base_temperature[:,1])

        # Calculate credible intervals for base model projections *using model indices that did not cause errors.
        ci_temperature = get_confidence_interval(collect(1765:end_year), base_temperature[good_indices,:], ci_interval_1, ci_interval_2)
        ci_co2         = get_confidence_interval(collect(1765:end_year), base_co2[good_indices,:],         ci_interval_1, ci_interval_2)
        ci_ocean_heat  = get_confidence_interval(collect(1765:end_year), base_ocean_heat[good_indices,:],  ci_interval_1, ci_interval_2)
        ci_oceanco2    = get_confidence_interval(collect(1765:end_year), base_oceanco2[good_indices,:],    ci_interval_1, ci_interval_2)
        ci_ch4         = get_confidence_interval(collect(1765:end_year), base_ch4[good_indices,:],         ci_interval_1, ci_interval_2)

        return base_temperature, base_co2, base_ch4, base_ocean_heat, base_oceanco2, pulse_temperature, pulse_co2,
               ci_temperature, ci_co2, ci_ocean_heat, ci_oceanco2, ci_ch4,
               error_indices, good_indices, ecs_sample
    end

    # Return function with user model specifications.
    return sneasych4_ecs
end
