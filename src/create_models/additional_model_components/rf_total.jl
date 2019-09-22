#--------------------------
# Total Raditiave Forcing.
#--------------------------

@defcomp rf_total begin

    α             = Parameter()             # Scaling factor to capture uncertainty in total aerosol radiative forcing.
    rf_O₃         = Parameter(index=[time]) # Radiative forcing from tropospheric ozone (Wm⁻²).
    rf_CO₂        = Parameter(index=[time]) # Radiative forcing from carbon dioxide (Wm⁻²).
    rf_CH₄        = Parameter(index=[time]) # Radiative forcing from methane (Wm⁻²).
    rf_CH₄_H₂O    = Parameter(index=[time]) # Radiative forcing from stratospheric water vapor due to methane oxidiation (Wm⁻²).
    rf_aerosol    = Parameter(index=[time]) # Radiative forcing from aerosols (Wm⁻²).
    rf_exogenous  = Parameter(index=[time]) # Radiative forcing from other (exogenous) sources (Wm⁻²).

    total_forcing = Variable(index=[time])  # Total radiative forcing from all sources (Wm⁻²).


    function run_timestep(p, v, d, t)

        if is_first(t)
            # Set initial radiative forcing to 0.
            v.total_forcing[t] = 0.0
        else
            # Calculate total radiative forcing (with aerosol contribution scaled by α).
            v.total_forcing[t] = p.rf_CO₂[t] + p.rf_CH₄[t] + p.rf_O₃[t] + p.rf_CH₄_H₂O[t] + (p.α * p.rf_aerosol[t]) + p.rf_exogenous[t]
        end
    end
end
