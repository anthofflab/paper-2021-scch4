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


#############################################################################################################################
# RUN BASELINE VERSION OF SNEASY+CH4.
#############################################################################################################################
# Description: This creates a function that runs two baseline versions of SNEASY+CH4 (a standard run, and a run with
#              an extra pulse of CH₄ emissions in a user-specified year) and saves the key model projection output.
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

function construct_sneasych4_baseline_case(climate_model::Symbol, rcp::String,  pulse_year::Int, pulse_size::Float64, end_year::Int)

    #------------------------------------------
    # Load and clean up forcing scenario data.
    #------------------------------------------

    # Load RCP scenario emissions data.
    rcp_emissions = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp*"_emissions.csv"), skiplines_begin=36))

    # Crop emissions to proper time periods (1765-end_year).
    rcp_indices   = findall((in)(collect(1765:end_year)), rcp_emissions.YEARS)
    rcp_emissions = rcp_emissions[rcp_indices, :]

    # Get years the model is run for (initializing it in 1765).
    model_years  = collect(1765:end_year)
    number_years = length(model_years)

    # Get indices for 1861-1880 to normalize temperature projections.
    indices_1861_1880 = findall((in)(collect(1861:1880)), model_years)

    # Load calibration data from 1765-2017 (measurement errors used in simulated noise).
    calibration_data = load_calibration_data(2017)

    # Pre-allocate vectors to hold simulated CAR(1) & AR(1) with measurement error noise.
    norm_oceanco2    = zeros(number_years)
    ar1_temperature  = zeros(number_years)
    ar1_oceanheat    = zeros(number_years)
    car1_co2         = zeros(number_years)
    car1_ch4         = zeros(number_years)

    # Replicate errors for years without observations over model time horizon.
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


    #---------------------------------------------------------------------------------------------------------------------
    # Create functions to set parameters and return results that differ across the four methane cycle models.
    #---------------------------------------------------------------------------------------------------------------------

    # Create a function to set the uncertain methane cycle parameters.
    # Note: This function assumes parameters follow the same ordering as in the `create_log_posterior.jl` file.
    update_ch4_params! = create_update_ch4_function(climate_model)

    # Create a function to return projected CH₄ concentrations (component and variable names differ across models).
    get_ch4_results! = create_get_ch4_results_function(climate_model)


    #---------------------------------------------------------------------------------------------------------------------
    # Given user-specified settings, create a function to run SNEASY+CH4 over the calibrated posterior model parameters.
    #---------------------------------------------------------------------------------------------------------------------

    function sneasych4_base_case(calibrated_parameters::Array{Float64,2}, ci_interval_1::Float64, ci_interval_2::Float64)

        # Calculate number of calibrated parameter samples (row = sample from joint posterior distribution, column = specific parameter).
        number_samples = size(calibrated_parameters, 1)

        # Pre-allocate arrays to store SNEASY+CH4 results.
        base_temperature  =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_co2          =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_ch4          =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_ocean_heat   =   zeros(Union{Missing, Float64}, number_samples, number_years)
        base_oceanco2     =   zeros(Union{Missing, Float64}, number_samples, number_years)

        pulse_temperature =   zeros(Union{Missing, Float64}, number_samples, number_years)
        pulse_co2         =   zeros(Union{Missing, Float64}, number_samples, number_years)

        # For each calibrated parameter sample, run the base and pulse versions of SNEASY+CH4 and store results.
        for i in 1:number_samples

            σ_temperature      = calibrated_parameters[i,1]
            σ_ocean_heat       = calibrated_parameters[i,2]
            σ²_white_noise_CO₂ = calibrated_parameters[i,3]
            σ²_white_noise_CH₄ = calibrated_parameters[i,4]
            ρ_temperature      = calibrated_parameters[i,5]
            ρ_ocean_heat       = calibrated_parameters[i,6]
            α₀_CO₂             = calibrated_parameters[i,7]
            α₀_CH₄             = calibrated_parameters[i,8]
            temperature_0      = calibrated_parameters[i,9]
            ocean_heat_0       = calibrated_parameters[i,10]
            CO₂_0              = calibrated_parameters[i,11]
            CH₄_0              = calibrated_parameters[i,12]
            N₂O_0              = calibrated_parameters[i,13]
            ECS                = calibrated_parameters[i,14]
            heat_diffusivity   = calibrated_parameters[i,15]
            rf_scale_aerosol   = calibrated_parameters[i,16]
            rf_scale_CH₄       = calibrated_parameters[i,17]
            F2x_CO₂            = calibrated_parameters[i,18]
            Q10                = calibrated_parameters[i,19]
            CO₂_fertilization  = calibrated_parameters[i,20]
            CO₂_diffusivity    = calibrated_parameters[i,21]

            # Set parameters for base version of SNEASY+CH4.
            update_param!(sneasych4_base, :t2co, ECS)
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
            update_ch4_params!(sneasych4_base, Vector(calibrated_parameters[i,:]))

            # Set parameters for version of SNEASY+CH4 with CH₄ emissions pulse.
            update_param!(sneasych4_pulse, :t2co, ECS)
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
            update_ch4_params!(sneasych4_pulse, Vector(calibrated_parameters[i,:]))

            # Add a check for cases where extreme parameter samples cause a model error.
            try

                # Run both models.
                run(sneasych4_base)
                run(sneasych4_pulse)

                # Create noise to superimpose on results using calibrated statistical parameters and measurement noise (note: both models use same estimated noise).
                ar1_temperature[:] = simulate_ar1_noise(number_years, σ_temperature, ρ_temperature, obs_error_temperature)
                ar1_oceanheat[:]   = simulate_ar1_noise(number_years, σ_ocean_heat,  ρ_ocean_heat,  obs_error_oceanheat)
                norm_oceanco2[:]   = rand(Normal(0,0.4*sqrt(10)), number_years)

                # CO₂ and CH₄ use CAR(1) statistical process parameters.
                car1_co2[:] = simulate_car1_noise(number_years, α₀_CO₂, σ²_white_noise_CO₂, obs_error_co2)
                car1_ch4[:] = simulate_car1_noise(number_years, α₀_CH₄, σ²_white_noise_CH₄, obs_error_ch4)

                # Store model projections resulting from parameter sample `i` for base model.
                base_temperature[i,:]  = sneasych4_base[:doeclim, :temp] .+ ar1_temperature .+ temperature_0
                base_co2[i,:]          = sneasych4_base[:ccm, :atmco2] .+ car1_co2
                base_ocean_heat[i,:]   = sneasych4_base[:doeclim, :heat_mixed] .+ sneasych4_base[:doeclim, :heat_interior] .+ ar1_oceanheat .+ ocean_heat_0
                base_oceanco2[i,:]     = sneasych4_base[:ccm, :atm_oc_flux] .+ norm_oceanco2
                base_ch4[i,:]          = get_ch4_results!(sneasych4_base) .+ car1_ch4

                # Store tempeature and CO₂ projections resulting from parameter sample `i` for pulse model (used for estimating the SC-CH₄).
                pulse_temperature[i,:] = sneasych4_pulse[:doeclim, :temp] .+ ar1_temperature
                pulse_co2[i,:]         = sneasych4_pulse[:ccm, :atmco2] .+ car1_co2

                # Normalize temperatures to be relative to the 1861-1880 mean.
                base_temperature[i,:]  = base_temperature[i,:]  .- mean(base_temperature[i, indices_1861_1880])
                pulse_temperature[i,:] = pulse_temperature[i,:] .- mean(pulse_temperature[i, indices_1861_1880])

            catch

                # Set values to -99999.99 if non-physical parameter samples produce a model error.
                base_temperature[i,:]  .= -99999.99
                base_co2[i,:]          .= -99999.99
                base_ocean_heat[i,:]   .= -99999.99
                base_oceanco2[i,:]     .= -99999.99
                base_ch4[i,:]          .= -99999.99
                pulse_temperature[i,:] .= -99999.99
                pulse_co2[i,:]         .= -99999.99
            end
        end

        # Calculate indices for runs that produce a model error.
        error_indices = findall(x-> x == -99999.99, base_temperature[:,1])
        good_indices  = findall(x-> x != -99999.99, base_temperature[:,1])

        # Calculate credible intervals for base model projections that did not produce a model error
        ci_temperature = get_confidence_interval(collect(1765:end_year), base_temperature[good_indices,:], ci_interval_1, ci_interval_2)
        ci_co2         = get_confidence_interval(collect(1765:end_year), base_co2[good_indices,:],         ci_interval_1, ci_interval_2)
        ci_ocean_heat  = get_confidence_interval(collect(1765:end_year), base_ocean_heat[good_indices,:],  ci_interval_1, ci_interval_2)
        ci_oceanco2    = get_confidence_interval(collect(1765:end_year), base_oceanco2[good_indices,:],    ci_interval_1, ci_interval_2)
        ci_ch4         = get_confidence_interval(collect(1765:end_year), base_ch4[good_indices,:],         ci_interval_1, ci_interval_2)

        return base_temperature, base_co2, base_ch4, base_ocean_heat, base_oceanco2, pulse_temperature, pulse_co2,
               ci_temperature, ci_co2, ci_ocean_heat, ci_oceanco2, ci_ch4,
               error_indices, good_indices
    end

    # Return function with user model specifications.
    return sneasych4_base_case
end
