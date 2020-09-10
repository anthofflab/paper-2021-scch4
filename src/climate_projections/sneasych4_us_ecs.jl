using CSVFiles
using DataFrames
using Distributions
using Mimi
using Statistics

#############################################################################################################################
# RUN VERSION OF SNEASY+CH4 SAMPLING FROM EQUILIBRIUM CLIMATE SENSITIVITY DISTRIBUTION USED FOR U.S. SC-CH4 ESTIMATES.
#############################################################################################################################
# Description: This creates a function that runs two versions of SNEASY+CH4 (a standard run, and a run with an extra pulse
#              of CH₄ emissions in a user-specified year) and saves the key model projection output. It samples equilibrium
#              climate sensitivity values from the distribution used to calculate official U.S. SC-CH4 estimates. All other
#              uncertain model, initial condition, and statistical process parameters fixed at their mean posterior values.
#
# Function Arguments:
#
#       climate_model = A symbol identifying the specific version of SNEASY+CH4 (options are :sneasy_fair, :sneasy_fund,
#                       :sneasy_hector, and :sneasy_magicc).
#       rcp           = A string identifying the RCP emissions and forcing scenario to use (options are "RCP26" and "RCP85").
#       pulse_year    = The year to add a pulse of methane emissions.
#       pulse_size    = The size of the methane emissions pulse in MtCH₄.
#       end_year      = The final year to run the model for.
#----------------------------------------------------------------------------------------------------------------------------

function construct_sneasych4_ecs(climate_model::Symbol, rcp::String,  pulse_year::Int, pulse_size::Float64, end_year::Int)

    #-------------------------------------------
    # Load and clean up forcing scenario data.
    #-------------------------------------------

    # Load RCP scenario emissions data.
    rcp_emissions = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp*"_emissions.csv"), skiplines_begin=36))

    # Crop emissions to proper time periods (1765-end_year).
    rcp_indices = findall((in)(collect(1765:end_year)), rcp_emissions.YEARS)
    rcp_emissions = rcp_emissions[rcp_indices, :]

    # Get years the model is run for (initializing it in 1765).
    model_years  = collect(1765:end_year)
    number_years = length(model_years)

    # Get indices for 1861-1880 to normalize temperature projections.
    indices_1861_1880 = findall((in)(collect(1861:1880)), model_years)

    # Load calibration data from 1765-2017 (measurement errors used in simulated noise).
    calibration_data = load_calibration_data(2017)

    # Pre-allocate vectors to hold simulated CAR(1) & AR(1) with measurement error noise.
    norm_oceanco2   = zeros(number_years)
    ar1_temperature = zeros(number_years)
    ar1_oceanheat   = zeros(number_years)
    car1_co2        = zeros(number_years)
    car1_ch4        = zeros(number_years)

    # Replicate errors for years without observations over model time horizon (note, CO₂ and CH₄ ice-core have constant error estimates).
    obs_error_temperature = replicate_errors(1765, end_year, calibration_data.hadcrut_temperature_sigma)
    obs_error_oceanheat   = replicate_errors(1765, end_year, calibration_data.ocean_heat_sigma)
    # Use constant Law Dome CH₄ observation errors for start period to 1983 (after which time-varying NOAA flask measurements start).
    obs_error_ch4 = replicate_errors(1765, end_year, calibration_data.noaa_ch4_sigma)
    obs_error_ch4[1:findfirst(isequal(1983), model_years)] .= unique(skipmissing(calibration_data.lawdome_ch4_sigma))[1]
    # Set constant CO₂ observations errors for start year-1958 (Law Dome), and 1959-end year (Mauna Loa).
    obs_error_co2 = ones(number_years) .* unique(skipmissing(calibration_data.maunaloa_co2_sigma))[1]
    obs_error_co2[1:findfirst(isequal(1958), model_years)] .= unique(skipmissing(calibration_data.lawdome_co2_sigma))[1]

    # Set up marginal CH₄ emissions time series to have an extra emission pulse in a user-specified year.
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

    # F2x_CO₂ has a default value in the component definition and therefore no
    # parameter in the model is created. The following line creates a parameter
    # in the model that can then be updated with update_param! in the inner loop.
    set_param!(sneasych4_base, :doeclim, :F2x_CO₂, 3.7)
    set_param!(sneasych4_pulse, :doeclim, :F2x_CO₂, 3.7)

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

        # Calculate number of equilibrium climate sensitivity samples.
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
        σ_temperature      = mean_posterior_parameters[1]
        σ_ocean_heat       = mean_posterior_parameters[2]
        σ²_white_noise_CO₂ = mean_posterior_parameters[3]
        σ²_white_noise_CH₄ = mean_posterior_parameters[4]
        ρ_temperature      = mean_posterior_parameters[5]
        ρ_ocean_heat       = mean_posterior_parameters[6]
        α₀_CO₂             = mean_posterior_parameters[7]
        α₀_CH₄             = mean_posterior_parameters[8]
        temperature_0      = mean_posterior_parameters[9]
        ocean_heat_0       = mean_posterior_parameters[10]
        CO₂_0              = mean_posterior_parameters[11]
        CH₄_0              = mean_posterior_parameters[12]
        N₂O_0              = mean_posterior_parameters[13]
        heat_diffusivity   = mean_posterior_parameters[15]
        rf_scale_aerosol   = mean_posterior_parameters[16]
        rf_scale_CH₄       = mean_posterior_parameters[17]
        F2x_CO₂            = mean_posterior_parameters[18]
        Q10                = mean_posterior_parameters[19]
        CO₂_fertilization  = mean_posterior_parameters[20]
        CO₂_diffusivity    = mean_posterior_parameters[21]

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

            # Wrap code in a try/catch statement in case extreme ECS values produce a model error.
            try

                # Run both models.
                run(sneasych4_base)
                run(sneasych4_pulse)

                # Create noise to superimpose on results using mean posterior statistical parameters and measurement noise (note: Both models use same estimated noise).
                ar1_temperature[:] = simulate_ar1_noise(number_years, σ_temperature, ρ_temperature, obs_error_temperature)
                ar1_oceanheat[:]   = simulate_ar1_noise(number_years, σ_ocean_heat,  ρ_ocean_heat,  obs_error_oceanheat)
                norm_oceanco2[:]   = rand(Normal(0,0.4*sqrt(10)), number_years)

                # CO₂ and CH₄ use CAR(1) statistical process parameters.
                car1_co2[:] = simulate_car1_noise(number_years, α₀_CO₂, σ²_white_noise_CO₂, obs_error_co2)
                car1_ch4[:] = simulate_car1_noise(number_years, α₀_CH₄, σ²_white_noise_CH₄, obs_error_ch4)

                # Store model projections resulting from parameter sample `i` for base model.
                base_temperature[i,:]  = sneasych4_base[:doeclim, :temp] .- mean(sneasych4_base[:doeclim, :temp][indices_1861_1880]) .+ ar1_temperature .+ temperature_0
                base_co2[i,:]          = sneasych4_base[:ccm, :atmco2] .+ car1_co2
                base_ocean_heat[i,:]   = sneasych4_base[:doeclim, :heat_interior] .+ ar1_oceanheat .+ ocean_heat_0
                base_oceanco2[i,:]     = sneasych4_base[:ccm, :atm_oc_flux] .+ norm_oceanco2
                base_ch4[i,:]          = get_ch4_results!(sneasych4_base) .+ car1_ch4

                # Store tempeature and CO₂ projections resulting from parameter sample `i` for pulse model (used for estimating the SC-CH₄).
                pulse_temperature[i,:] = sneasych4_pulse[:doeclim, :temp] .- mean(sneasych4_pulse[:doeclim, :temp][indices_1861_1880]) .+ ar1_temperature .+ temperature_0
                pulse_co2[i,:]         = sneasych4_pulse[:ccm, :atmco2] .+ car1_co2

            catch

                # Set values to -99999.99 if non-physical results produce a model error.
                base_temperature[i,:]  .= -99999.99
                base_co2[i,:]          .= -99999.99
                base_ocean_heat[i,:]   .= -99999.99
                base_oceanco2[i,:]     .= -99999.99
                base_ch4[i,:]          .= -99999.99
                pulse_temperature[i,:] .= -99999.99
                pulse_co2[i,:]         .= -99999.99
            end
        end

        # Identify model indices that caused a model error or yield non-physical outcomes (i.e. strongly negative temperatures in 2300 under RCP 8.5).
        error_indices = findall(x-> x < -10.0, base_temperature[:,end])
        good_indices  = findall(!in(error_indices), collect(1:number_samples))

        # Calculate credible intervals for base model projections using model indices that did not cause errors.
        ci_temperature = get_confidence_interval(collect(1765:end_year), base_temperature[good_indices,:], ci_interval_1, ci_interval_2)
        ci_co2         = get_confidence_interval(collect(1765:end_year), base_co2[good_indices,:],         ci_interval_1, ci_interval_2)
        ci_ocean_heat  = get_confidence_interval(collect(1765:end_year), base_ocean_heat[good_indices,:],  ci_interval_1, ci_interval_2)
        ci_oceanco2    = get_confidence_interval(collect(1765:end_year), base_oceanco2[good_indices,:],    ci_interval_1, ci_interval_2)
        ci_ch4         = get_confidence_interval(collect(1765:end_year), base_ch4[good_indices,:],         ci_interval_1, ci_interval_2)

        return base_temperature[good_indices,:], base_co2[good_indices,:], base_ch4[good_indices,:], base_ocean_heat[good_indices,:], base_oceanco2[good_indices,:], pulse_temperature[good_indices,:], pulse_co2[good_indices,:],
               ci_temperature, ci_co2, ci_ocean_heat, ci_oceanco2, ci_ch4,
               error_indices, good_indices, ecs_sample
    end

    # Return function with user model specifications.
    return sneasych4_ecs
end
