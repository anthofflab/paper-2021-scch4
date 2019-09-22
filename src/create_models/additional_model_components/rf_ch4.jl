# ----------------------------------------------------------------
# Radiative forcing from methane (Etminan et al. 2016 equations).
# ----------------------------------------------------------------

@defcomp rf_ch4_etminan begin

    a₃         = Parameter()             # Methane forcing constant.
    b₃         = Parameter()             # Methane forcing constant.
    N₂O_0      = Parameter()             # Initial (pre-industrial) nitrous oxide concentration (ppb).
    scale_CH₄  = Parameter()             # Scaling factor for uncertainty in methane radiative forcing.
    CH₄_0      = Parameter()             # Initial (pre-industrial) methane concentration (ppb).
    N₂O        = Parameter(index=[time]) # Atmospheric nitrous oxide concentration (ppb).
    CH₄        = Parameter(index=[time]) # Atmospheric methane concentration (ppb).

    rf_CH₄     = Variable(index=[time])  # Direct forcing from atmospheric methane concentrations (Wm⁻²).


    function run_timestep(p, v, d, t)

        # Create temporary averaging variables.
        M̄ = 0.5 * (p.CH₄[t] + p.CH₄_0)
        N̄ = 0.5 * (p.N₂O[t] + p.N₂O_0)

        # Direct methane radiative forcing.
        v.rf_CH₄[t] = ((p.a₃ * M̄ + p.b₃ * N̄ + 0.043) * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))) * p.scale_CH₄
    end
end
