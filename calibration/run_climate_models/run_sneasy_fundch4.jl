#-------------------------------------------------------------------------------------
# Create a function to run SNEASY-FUND and return model output used in calibration.
#-------------------------------------------------------------------------------------

# Load model file.
include("../../src/create_models/create_sneasy_fundch4.jl")

function construct_run_sneasy_fundch4(calibration_end_year::Int)

    # Load an instance of SNEASY-FUND.
    m = create_sneasy_fundch4(start_year=1765, end_year=calibration_end_year, etminan_ch4_forcing=true)

    # Get indices needed to normalize temperature anomalies relative to 1861-1880 mean.
    index_1861, index_1880 = findall((in)([1861, 1880]), collect(1765:calibration_end_year))

    # Given user settings, create a function to run SNEASY-FUND and return model output used for calibration.
    function run_sneasy_fundch4!(
        params::Array{Float64,1},
        modeled_CO₂::Vector{Float64},
        modeled_CH₄::Vector{Float64},
        modeled_oceanCO₂_flux::Vector{Float64},
        modeled_temperature::Vector{Float64},
        modeled_ocean_heat::Vector{Float64})

        # Assign names to uncertain model and initial condition parameters for convenience
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

        #----------------------------------------------------------
        # Set SNEASY-FUND to use sampled parameter values.
        #----------------------------------------------------------

        # ---- Diffusion Ocean Energy balance CLIMate model (DOECLIM) ---- #
        set_param!(m, :doeclim, :t2co, ECS)
        set_param!(m, :doeclim, :kappa, heat_diffusivity)
        set_param!(m, :doeclim, :F2x_CO₂, F2x_CO₂)

        # ---- Carbon Cycle ---- #
        set_param!(m, :ccm, :Q10, Q10)
        set_param!(m, :ccm, :Beta, CO₂_fertilization)
        set_param!(m, :ccm, :Eta, CO₂_diffusivity)
        update_param!(m, :atmco20, CO₂_0)

        # ---- Methane Cycle ---- #
        set_param!(m, :climatech4cycle, :ch4pre, CH₄_0)
        set_param!(m, :climatech4cycle, :acch4_0, CH₄_0)
        set_param!(m, :climatech4cycle, :lifech4, τ_troposphere)

        # ---- Direct Methane Radiative Forcing ---- #
        set_param!(m, :rf_ch4_etminan, :CH₄_0, CH₄_0)
        set_param!(m, :rf_ch4_etminan, :N₂O_0, N₂O_0)
        set_param!(m, :rf_ch4_etminan, :scale_CH₄, rf_scale_CH₄)

        # ---- Total Methane Radiative Forcing (including indirect effects) ---- #
        set_param!(m, :rf_ch4_total_fund, :CH₄_0, CH₄_0)

        # ---- Carbon Dioxide Radiative Forcing ---- #
        set_param!(m, :rf_co2_etminan, :CO₂_0, CO₂_0)
        set_param!(m, :rf_co2_etminan, :N₂O_0, N₂O_0)
        set_param!(m, :rf_co2_etminan, :rf_scale_CO₂, co2_rf_scale(F2x_CO₂, CO₂_0, N₂O_0))

        # ---- Total Radiative Forcing ---- #
        set_param!(m, :rf_total, :α, rf_scale_aerosol)

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
        modeled_CH₄[:] = m[:climatech4cycle, :acch4]

        return
    end

    # Return function.
    return run_sneasy_fundch4!
end
