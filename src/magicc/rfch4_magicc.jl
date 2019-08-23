using Mimi

@defcomp rfch4magicc begin
    a₃      = Parameter()             # CH₄ forcing constant.
    b₃      = Parameter()             # CH₄ forcing constant.
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    N₂O     = Parameter(index=[time]) # N₂O concentration (ppb).
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)

    STRATH2O        = Parameter() #Ratio DELQH2O/DELQCH4, for Strat H2O from CH4 Oxidation
    TROZSENS        = Parameter() #TROP OZONE FORCING SENSITIVITY (TAR=0.042, EMPIRICAL=0.0389)
    OZCH4           = Parameter() #CH4 TERM FACTOR IN OZONE EQU
    CH4_0           = Parameter()
    CH4             = Parameter(index=[time]) #CH4 concentration
    QCH4OZ          = Variable(index=[time]) #Enhancement of QCH4 due to CH4-induced ozone production.
    QCH4H2O         = Variable(index=[time]) #Additional indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation
    QMeth           = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only

end

function run_timestep(s::rfch4magicc, t::Int)
    v = s.Variables
    p = s.Parameters

    # Create temporary averaging variables.
    M_hat = 0.5 * (p.CH4[t] + p.CH4_0)
    N_hat = 0.5 * (p.N₂O[t] + p.N₂O_0)

    #Direct CH4 radiative forcing (from Myhre, Highwood, & Shine; GRL 2016).
    v.QMeth[t] = ((p.a₃ * M_hat + p.b₃ * N_hat + 0.043) * (sqrt(p.CH4[t]) - sqrt(p.CH4_0))) * p.scale_CH₄

    #Calculate indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation (based on direct CH4 forcing without overlap formula from MAGICC).
    v.QCH4H2O[t] = p.STRATH2O * 0.036 * (sqrt(p.CH4[t]) - sqrt(p.CH4_0))

    #Calculate enhnacement of direct CH4 forcing due to ch4-induced ozone production
    #NOTE: Old concentration based equation added 2000 O₃ focring due to CH₄. This term was removed for emissions version.
    v.QCH4OZ[t] = p.TROZSENS * p.OZCH4 * log(p.CH4[t]/p.CH4_0)

end
