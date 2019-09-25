# -----------------------------------------------------------------------
# Radiative Forcing From Methane (based on Myhre et al. equations).
# -----------------------------------------------------------------------

# Create function to account for overlap in CH₄ and N₂O absoprtion bands.
function overlap(M, N)
    d = 1.0 + 2.01e-5*(M*N)^0.75 + 5.31e-15*M*(M*N)^1.52
    return 0.47 * log(d)
end


@defcomp rf_ch4_myhre_fair begin

    N₂O_0           = Parameter()             # Initial (pre-industrial) nitrous oxide concentration (ppb).
    CH₄_0           = Parameter()             # Initial (pre-industrial) methane concentration (ppb).
    scale_CH₄       = Parameter()             # Scaling factor for uncertainty in methane radiative forcing.
    H₂O_share       = Parameter()             # Share of direct methane forcing used to estimate stratospheric water vapor forcing due to methane oxidation.
    CH₄             = Parameter(index=[time]) # Atmospheric methane concentration (ppb).

    forcing_CH₄     = Variable(index=[time])  # Radiative forcing from methane (Wm⁻²).
    forcing_CH₄_H₂O = Variable(index=[time])  # Radiative forcing from stratospheric water vapor due to methane oxidiation (Wm⁻²).


    function run_timestep(p, v, d, t)

        if is_first(t)
            # Set initial radiative forcing to 0.
            v.forcing_CH₄[t] = 0.0
            v.forcing_CH₄_H₂O[t] = 0.0

        else

            # Calculate direct CH₄ radiative forcing.
            direct_rf = 0.036 * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0)) - (overlap(p.CH₄[t], p.N₂O_0) - overlap(p.CH₄_0, p.N₂O_0))

            # Calculate direct radiative forcing from CH4, accounting for forcing uncertainty.
            v.forcing_CH₄[t] = direct_rf * p.scale_CH₄

            #Calculate indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation.
            v.forcing_CH₄_H₂O[t] = direct_rf * p.H₂O_share
        end
    end
end
