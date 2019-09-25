# -----------------------------------------------------------------------
# Direct Radiative Forcing From Methane (based on FUND model equations).
# -----------------------------------------------------------------------

# Create function to account for overlap in CH₄ and N₂O absoprtion bands.
function interact(M, N)
    d = 1.0 + (M * N)^0.75 * 2.01E-5 + (M * N)^1.52 * M * 5.31E-15
    return 0.47 * log(d)
end


@defcomp rf_ch4_direct_fund begin

    CH₄_0         = Parameter()             # Initial (pre-industrial) methane concentration (ppb).
    N₂O_0         = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    scale_CH₄     = Parameter()             # Scaling factor for uncertainty in methane radiative forcing.
    CH₄           = Parameter(index=[time]) # Atmospheric methane concetration for current period (ppb).

    rf_ch4_direct = Variable(index=[time])  # Direct forcing from atmospheric methane concentrations (Wm⁻²).


    function run_timestep(p, v, d, t)
        if is_first(t)
            # Set initial radiative forcing to 0.
            v.rf_ch4_direct[t] = 0.0
        else
            # Create temporary variable accounting for overlap between pre-industrial CH₄ and N₂O (for convenience).
            ch4n2o = interact(p.CH₄_0, p.N₂O_0)

            # Calculate direct CH₄ radiative forcing.
            v.rf_ch4_direct[t] = (0.036 * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0)) - interact(p.CH₄[t], p.N₂O_0) + ch4n2o) * p.scale_CH₄
        end
    end
end
