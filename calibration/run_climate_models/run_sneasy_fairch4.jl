#-------------------------------------------------------------------------------------
# Create a function to run SNEASY-FAIR and return model output used in calibration.
#-------------------------------------------------------------------------------------

function construct_run_sneasy_fairch4(calibration_end_year::Int)

    # Load an instance of SNEASY-FAIR.
    m = create_sneasy_fairch4(start_year=1765, end_year=calibration_end_year, etminan_ch4_forcing=true)

    # F2x_CO₂ has a default value in the component definition and therefore no
    # parameter in the model is created. The following line creates a parameter
    # in the model that can then be updated with update_param! in the inner loop.
    set_param!(m, :doeclim, :F2x_CO₂, 3.7)

    # Get indices needed to normalize temperature anomalies relative to 1861-1880 mean.
    index_1861, index_1880 = findall((in)([1861, 1880]), collect(1765:calibration_end_year))

    # Calculate number of time periods to run model for.
    n_steps = length(1765:calibration_end_year)

    # Given user settings, create a function to run SNEASY-FAIR and return model output used for calibration.
    function run_sneasy_fairch4!(
        params::Array{Float64,1},
        modeled_CO₂::Vector{Float64},
        modeled_CH₄::Vector{Float64},
        modeled_oceanCO₂_flux::Vector{Float64},
        modeled_temperature::Vector{Float64},
        modeled_ocean_heat::Vector{Float64})

        # Assign names to uncertain model and initial condition parameters for convenience.
        # Note: This assumes "params" is the full vector of uncertain parameters with the same ordering as in "create_log_posterior.jl".
        temperature_0     = params[9]
        ocean_heat_0      = params[10]
        CO₂_0             = params[11]
        CH₄_0             = params[12]
        N₂O_0             = params[13]
        ECS               = params[14]
        heat_diffusivity  = params[15]
        rf_scale_aerosol  = params[16]
        rf_scale_CH₄      = params[17]
        F2x_CO₂           = params[18]
        Q10               = params[19]
        CO₂_fertilization = params[20]
        CO₂_diffusivity   = params[21]
        τ_troposphere     = params[22]
        CH₄_natural       = params[23]

        #----------------------------------------------------------
        # Set SNEASY-FAIR to use sampled parameter values.
        #----------------------------------------------------------

        # ---- Diffusion Ocean Energy balance CLIMate model (DOECLIM) ---- #
        update_param!(m, :t2co, ECS)
        update_param!(m, :kappa, heat_diffusivity)
        update_param!(m, :F2x_CO₂, F2x_CO₂)

        # ---- Carbon Cycle ---- #
        update_param!(m, :Q10, Q10)
        update_param!(m, :Beta, CO₂_fertilization)
        update_param!(m, :Eta, CO₂_diffusivity)
        update_param!(m, :atmco20, CO₂_0)

        # ---- Methane Cycle ---- #
        update_param!(m, :natural_emiss_CH₄, ones(n_steps) .* CH₄_natural)
        update_param!(m, :τ_CH₄, τ_troposphere)

        # ---- Methane Radiative Forcing ---- #
        update_param!(m, :scale_CH₄, rf_scale_CH₄)

        # ---- Carbon Dioxide Radiative Forcing ---- #
        update_param!(m, :CO₂_0, CO₂_0)
        update_param!(m, :rf_scale_CO₂, co2_rf_scale(F2x_CO₂, CO₂_0, N₂O_0))

        # ---- Total Radiative Forcing ---- #
        update_param!(m, :α, rf_scale_aerosol)

        # ---- Shared parameters ---- #
        update_param!(m, :CH₄_0, CH₄_0)
        update_param!(m, :N₂O_0, N₂O_0)

        # Run model.
        run(m)

        #----------------------------------------------------------
        # Calculate model output being compared to observations.
        #----------------------------------------------------------

        # Atmospheric concentration of CO₂.
        modeled_CO₂[:] = m[:ccm, :atmco2]

        # Ocean carbon flux (Note: timesteps cause last `atm_oc_flux` value to equal `missing`, so exclude it here).
        modeled_oceanCO₂_flux[1:end-1] = m[:ccm, :atm_oc_flux][1:end-1]

        # Global surface temperature anomaly (normalized to 1861-1880 mean with initial condition offset).
        modeled_temperature[:] = m[:doeclim, :temp] .- mean(m[:doeclim, :temp][index_1861:index_1880]) .+ temperature_0

        # Ocean heat content (with initial condition offset).
        modeled_ocean_heat[:] = m[:doeclim, :heat_mixed] .+ m[:doeclim, :heat_interior] .+ ocean_heat_0

        # Atmospheric concentration of CH₄.
        modeled_CH₄[:] = m[:ch4_cycle, :CH₄]

        return
    end

    # Return function.
    return run_sneasy_fairch4!
end
