
@defcomp rfch4hector begin
    a₃      = Parameter()             # CH₄ forcing constant.
    b₃      = Parameter()             # CH₄ forcing constant.
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    N₂O     = Parameter(index=[time]) # N₂O concentration (ppb).
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)
    CH4_0           = Parameter()
    CH4             = Parameter(index=[time]) #Ch4 concentration ppbv in time t
    rf_CH4          = Variable(index=[time]) #Direct Radiative forcing for methane
end

#The function 'interact' accounts for the overlap in CH4 and N2O in their absoprtion bands
function interact(M, N)
    d = 1.0 + (M * N)^0.75 * 2.01E-5 + (M * N)^1.52 * M * 5.31E-15
    return 0.47 * log(d)
end

function run_timestep(s::rfch4hector, t::Int)
    v = s.Variables
    p = s.Parameters

    # Create temporary averaging variables.
    M_hat = 0.5 * (p.CH4[t] + p.CH4_0)
    N_hat = 0.5 * (p.N₂O[t] + p.N₂O_0)

    #Direct CH4 radiative forcing (from Myhre, Highwood, & Shine; GRL 2016).
    v.rf_CH4[t] = ((p.a₃ * M_hat + p.b₃ * N_hat + 0.043) * (sqrt(p.CH4[t]) - sqrt(p.CH4_0))) * p.scale_CH₄
end
