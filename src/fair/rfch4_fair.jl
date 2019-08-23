using Mimi

@defcomp rfch4fair begin
    a₃      = Parameter()             # CH₄ forcing constant.
    b₃      = Parameter()             # CH₄ forcing constant.
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    N₂O     = Parameter(index=[time]) # N₂O concentration (ppb).
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)

    H₂O_share       = Parameter() # Share of direct CH₄ forcing used to estimate stratospheric water vapor forcing due to CH₄ oxidation. (0.12 for etminan forcing equations)
    CH₄_0           = Parameter()
    CH₄             = Parameter(index=[time]) #CH4 concentration
    forcing_CH₄_H₂O = Variable(index=[time]) #Additional indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation
    forcing_CH₄     = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only
end

function run_timestep(s::rfch4fair, t::Int)
    v = s.Variables
    p = s.Parameters

    # Create temporary averaging variables.
    M_hat = 0.5 * (p.CH₄[t] + p.CH₄_0)
    N_hat = 0.5 * (p.N₂O[t] + p.N₂O_0)

    #Direct CH4 radiative forcing (from Myhre, Highwood, & Shine; GRL 2016).
    v.forcing_CH₄[t] = ((p.a₃ * M_hat + p.b₃ * N_hat + 0.043) * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))) * p.scale_CH₄

    #Calculate indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation.
    # NOTE: Removing scaling term for direct CH4 forcing here (CO2 is just from direct ch4 without uncertainty.)
    v.forcing_CH₄_H₂O[t] = p.H₂O_share *  (v.forcing_CH₄[t] / p.scale_CH₄)

end
