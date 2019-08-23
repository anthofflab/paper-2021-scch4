
@defcomp rfch4 begin
    a₃      = Parameter()             # CH₄ forcing constant.
    b₃      = Parameter()             # CH₄ forcing constant.
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    N₂O     = Parameter(index=[time]) # N₂O concentration (ppb).
    ch4ind = Parameter()            # Scaling term to represent indirect radiative forcing increase for CH4 in FUND model.
    CH₄             = Parameter(index=[time]) #CH4 concentration
    CH₄_0           = Parameter()
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)

    rf_ch4_direct           = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations 
    rf_ch4_indirect           = Variable(index=[time]) #indirect radiative forcing from atmospheric ch4 concentrations 
    rf_ch4_total                   = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only

end


function run_timestep(s::rfch4, t::Int)
    v = s.Variables
    p = s.Parameters

    # Create temporary averaging variables.
    M_hat = 0.5 * (p.CH₄[t] + p.CH₄_0)
    N_hat = 0.5 * (p.N₂O[t] + p.N₂O_0)

    #Direct CH4 radiative forcing (from Myhre, Highwood, & Shine; GRL 2016).
    v.rf_ch4_direct[t] = ((p.a₃ * M_hat + p.b₃ * N_hat + 0.043) * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))) * p.scale_CH₄

    #Indirect CH4 forcing from FUND (assume it remains as 40% of old RF equation in FUND, not accounting for CH4-N2O band overlap).
    v.rf_ch4_indirect[t] = 0.036 * p.ch4ind * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))

    # Calculate total forcing (sum of new direct forcing (shortwave) and FUND's indirect forcing equation)
    v.rf_ch4_total[t] = v.rf_ch4_direct[t] + v.rf_ch4_indirect[t]
end
