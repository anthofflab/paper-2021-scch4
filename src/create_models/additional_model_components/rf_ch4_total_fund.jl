# -----------------------------------------------------------------------
# Total Radiative Forcing From Methane (based on FUND model equations).
# -----------------------------------------------------------------------

@defcomp rf_ch4_total_fund begin

    CH₄_0           = Parameter()             # Initial (pre-industrial) methane concentration (ppb).
    ϕ               = Parameter()             # Scaling term to represent indirect radiative forcing from methane (25% increase from tropospheric ozone, 15% increase from stratospheric water vapor).
    CH₄             = Parameter(index=[time]) # Atmospheric methane concetration for current period (ppb).
    rf_ch4_direct   = Parameter(index=[time]) # Direct forcing from atmospheric methane concentrations (Wm⁻²).

    rf_ch4_indirect = Variable(index=[time])  # Indirect radiative forcing from atmospheric methane concentrations (Wm⁻²).
    rf_ch4_total    = Variable(index=[time])  # Total radiative forcing (direct and indirect effects) from methane (Wm⁻²).


    function run_timestep(p, v, d, t)

        if is_first(t)
            # Set initial radiative forcing to 0.
            v.rf_ch4_indirect[t] = 0.0
            v.rf_ch4_total[t] = 0.0
        else
            # Indirect CH₄ forcing from FUND (following FUND equations, this is 40% of direct forcing without accounting for CH₄-N₂O absorption overlap).
            v.rf_ch4_indirect[t] = 0.036 * p.ϕ * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))

            # Calculate total CH₄ radiative forcing.
            v.rf_ch4_total[t] = p.rf_ch4_direct[t] + v.rf_ch4_indirect[t]
        end
    end
end
