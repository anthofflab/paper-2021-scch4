using Mimi

@defcomp rfch4fair begin
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    H₂O_share       = Parameter() # Share of direct CH₄ forcing used to estimate stratospheric water vapor forcing due to CH₄ oxidation. (0.12 for etminan forcing equations)
    CH₄_0           = Parameter()
    CH₄             = Parameter(index=[time]) #Natural CH4 emissions (Tgrams)
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)
    forcing_CH₄_H₂O = Variable(index=[time]) #Additional indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation
    forcing_CH₄     = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only

end

#The function 'overlap' accounts for the overlap in CH4 and N2O in their absoprtion bands
function overlap(M, N)
    d = 1.0 + 2.01e-5*(M*N)^0.75 + 5.31e-15*M*(M*N)^1.52
    return 0.47 * log(d)
end

function run_timestep(s::rfch4fair, t::Int)
    v = s.Variables
    p = s.Parameters

    #direct radiative forcing from atmospheric ch4 concentrations only
    direct_rf = 0.036 * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0)) - (overlap(p.CH₄[t], p.N₂O_0) - overlap(p.CH₄_0, p.N₂O_0))

    # Calculate direct radiative forcing from CH4, accounting for uncertainty in forcing (assume uncertainty not on h2o).
    v.forcing_CH₄[t] = direct_rf * p.scale_CH₄

    #Calculate indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation.
    v.forcing_CH₄_H₂O[t] = direct_rf * p.H₂O_share

end
