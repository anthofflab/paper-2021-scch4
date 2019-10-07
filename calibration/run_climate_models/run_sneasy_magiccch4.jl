#-------------------------------------------------------------------------------------
# Create a function to run SNEASY-MAGICC and return model output used in calibration.
#-------------------------------------------------------------------------------------

# Load require model files
include("../../src/create_models/create_sneasy_magiccch4.jl")

# Create a function that will run SNEASY-MAGICC and return required model output.
function construct_run_sneasy_magiccch4(calibration_end_year::Int)

    # Load an instance of SNEASY-MAGICC (with option of how many years to calibrate model for).
    m = create_sneasy_magiccch4(start_year=1765, end_year=calibration_end_year, etminan_ch4_forcing=true)

    # Get indices needed to normalize temperature anomalues (currently using 1861-1880 mean).
    index_1861, index_1880 = findall((in)([1861, 1880]), collect(1765:calibration_end_year))

    function run_sneasy_magiccch4!(
        params::Array{Float64,1},
        modeled_CO₂::Vector{Float64},
        modeled_CH₄::Vector{Float64},
        modeled_oceanCO₂_flux::Vector{Float64},
        modeled_temperature::Vector{Float64},
        modeled_ocean_heat::Vector{Float64})

        # Assign names to uncertain model and initial condition parameters (assumes p = full vector of uncertain parameters).
        temperature_0     = params[12]
        ocean_heat_0      = params[13]
        CO₂_0             = params[14]
        CH₄_0             = params[15]
        N₂O_0             = params[16]
        ECS               = params[17]
        heat_diffusivity  = params[18]
        rf_scale_aerosol  = params[19]
        rf_scale_CH₄      = params[20]
        F2x_CO₂           = params[21]
        Q10               = params[22]
        CO₂_fertilization = params[23]
        CO₂_diffusivity   = params[24]
        τ_troposphere     = params[25]
        CH₄_natural       = params[26]
        τ_soil            = params[27]
        τ_stratosphere    = params[28]


        #----------------------------------------------------------
        # Set SNEASY-MAGICC with sampled parameter values.
        #----------------------------------------------------------

        # ---- Diffusion Ocean Energy balance CLIMate model (DOECLIM) ---- #
        set_param!(m, :doeclim, :t2co, ECS)
        set_param!(m, :doeclim, :kappa, heat_diffusivity)
        set_param!(m, :doeclim, :F2x_CO₂, F2x_CO₂)

        # ---- Carbon Cycle ---- #
        set_param!(m, :ccm, :Q10, Q10)
        set_param!(m, :ccm, :Beta, CO₂_fertilization)
        set_param!(m, :ccm, :Eta, CO₂_diffusivity)
        set_param!(m, :ccm, :atmco20, CO₂_0)

        # ---- Methane Cycle ---- #
        set_param!(m, :ch4_cycle, :CH₄_0, CH₄_0)
        set_param!(m, :ch4_cycle, :CH4_natural, CH₄_natural)
        set_param!(m, :ch4_cycle, :TAUSOIL, τ_soil)
        set_param!(m, :ch4_cycle, :TAUSTRAT, τ_stratosphere)
        set_param!(m, :ch4_cycle, :TAUINIT, τ_troposphere)

        # ---- Methane Radiative Forcing ---- #
        set_param!(m, :rf_ch4_etminan, :CH₄_0, CH₄_0)
        set_param!(m, :rf_ch4_etminan, :N₂O_0, N₂O_0)
        set_param!(m, :rf_ch4_etminan, :scale_CH₄, rf_scale_CH₄)

        # ---- Straospheric Water Vapor From Oxidized Methane Radiative Forcing ---- #
        set_param!(m, :rf_ch4h2o, :CH₄_0, CH₄_0)

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
        modeled_ocean_heat[:] = m[:doeclim, :heat_interior] .+ ocean_heat_0

        # Atmospheric concentration of CH₄ (exclude first 'missing' value not used in calibration).
        modeled_CH₄[2:end] = m[:ch4_cycle, :CH₄][2:end]

        return
    end

   return run_sneasy_magiccch4!
end
