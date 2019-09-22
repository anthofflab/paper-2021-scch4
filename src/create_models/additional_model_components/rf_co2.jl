# -----------------------------------------------------------------------
# Radiative forcing from carbon dioxide (Etminan et al. 2016 equations).
# -----------------------------------------------------------------------

@defcomp rf_co2_etminan begin

    a₁           = Parameter()             # Carbon dioxide forcing constant.
    b₁           = Parameter()             # Carbon dioxide forcing constant.
    c₁           = Parameter()             # Carbon dioxide forcing constant.
    CO₂_0        = Parameter()             # Initial (pre-industrial) carbon dioxide concentration (ppm).
    N₂O_0        = Parameter()             # Initial (pre-industrial) nitrous oxide concentration (ppb).
    rf_scale_CO₂ = Parameter()             # Radiative forcing scaling term (to keep forcing from 2x CO₂ consistent).
    N₂O          = Parameter(index=[time]) # Atmospheric nitrous oxide concentration (ppb).
    CO₂          = Parameter(index=[time]) # Atmospheric carbon dioxide concentration (ppm).

    rf_CO₂       = Variable(index=[time])  # Forcing from atmospheric carbon dioxide concentrations (Wm⁻²).


    function run_timestep(p, v, d, t)

        # Create temporary variables.
        CO₂_diff = p.CO₂[t]-p.CO₂_0
        N̄ = 0.5 * (p.N₂O[t] + p.N₂O_0)

        # Calculate carbon dioxide radiative forcing.
        v.rf_CO₂[t] = ((p.a₁*CO₂_diff^2 + p.b₁*abs(CO₂_diff) + p.c₁*N̄ + 5.36) * log(p.CO₂[t] / p.CO₂_0)) * p.rf_scale_CO₂
    end
end
