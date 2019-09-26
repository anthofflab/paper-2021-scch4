include("../../src/sneasy_magicc/sneasy_magicc.jl")

function construct_run_mimi_sneasy()
    m = getsneasy_magicc(start_year=1765, end_year=2017)
    
    # Indices for normalizing data (use 1861-1880 for temp following FAIR 1.3)
    index_1861, index_1880 = findin(collect(1765:2300), [1861, 1880])
    
    function run_mimi_sneasy!(
           model_co2::Vector{Float64},
           model_ch4::Vector{Float64},
           model_ocflux::Vector{Float64},
           model_temperature::Vector{Float64},
           model_ocheat::Vector{Float64},
           S::Float64,
           kappa::Float64,
           alpha::Float64,
           Q10::Float64,
           beta::Float64,
           eta::Float64,
           TAUINIT::Float64,
           CH4_Nat::Float64,     # Natural CH4 emissions
           Tsoil::Float64,      # Soil sink lifetime
           Tstrat::Float64,     # Start sink lifetime
           CO2_0::Float64,
           T0::Float64,
           H0::Float64,
           CH4_0::Float64,
           N2O_0::Float64,
           F2x_CO₂::Float64,
           rf_scale_CH₄::Float64)

        #Calculate CO₂ RF scaling based on F2x_CO₂ value.
        scale_CO₂ = co2_rf_scale(F2x_CO₂, CO2_0, N2O_0)

        setparameter(m, :doeclim, :t2co, S)
        setparameter(m, :doeclim, :kappa, kappa)
        #setparameter(m, :doeclim, :T0, T0)
        setparameter(m, :doeclim, :H0, H0)
        setparameter(m, :doeclim, :F2x_CO₂,  F2x_CO₂)

        setparameter(m, :ccm, :Q10, Q10)
        setparameter(m, :ccm, :Beta, beta)
        setparameter(m, :ccm, :Eta, eta)
        setparameter(m, :ccm, :atmco20, CO2_0)
        setparameter(m, :radiativeforcing, :alpha, alpha)

        setparameter(m, :rfch4magicc, :CH4_0, CH4_0)
        setparameter(m, :rfch4magicc, :N₂O_0, N2O_0)
        setparameter(m, :rfch4magicc, :scale_CH₄, rf_scale_CH₄)

        setparameter(m, :rfco2, :CO₂_0, CO2_0)
        setparameter(m, :rfco2, :N₂O_0, N2O_0)  #taken from 1765 RCP85 data
        setparameter(m, :rfco2, :scale_CO₂, scale_CO₂)

        setparameter(m, :ch4cyclemagicc, :TAUINIT, TAUINIT)
        setparameter(m, :ch4cyclemagicc, :CH4_0, CH4_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)
        #setparameter(m, :ch4cyclemagicc, :CH4_2, CH4_2)
        setparameter(m, :ch4cyclemagicc, :TAUSOIL, Tsoil)
        setparameter(m, :ch4cyclemagicc, :TAUSTRAT, Tstrat)
        setparameter(m, :ch4cyclemagicc, :CH4_natural, CH4_Nat)

        run(m)

        # Does not need to be normalized.
        model_co2[:] = m[:ccm, :atmco2]
        # Does not need to be normalized.
        model_ocflux[:] = m[:ccm, :atm_oc_flux]
        #Normalize to 1850-1870 mean
        #model_temperature[:] = m[:doeclim, :temp] .- m[:doeclim, :temp][index_1850] .+ T0
        model_temperature[:] = m[:doeclim, :temp] .- mean(m[:doeclim, :temp][index_1861:index_1880]) .+ T0
        # No need to normalize, using H0 initial condition/offset.
        #model_ocheat[:] = m[:doeclim, :heat_interior] + H0
        model_ocheat[:] = m[:doeclim, :heat_interior] .+ H0
        # CH4 concnetration (does not need to be normalized).
        model_ch4[:] = m[:ch4cyclemagicc, :CH4]

        return
    end

    return run_mimi_sneasy!
end
