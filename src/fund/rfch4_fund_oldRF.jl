
@defcomp rfch4 begin
    
    N₂O_0   = Parameter()             # Initial (pre-industrial) N₂O concentration (ppb).
    ch4ind = Parameter()            # Scaling term to represent indirect radiative forcing increase for CH4 in FUND model.
    CH₄             = Parameter(index=[time]) #CH4 concentration
    CH₄_0           = Parameter()
    scale_CH₄ = Parameter() #Scaling factor for uncertainty in RF (following FAIR 1.3 approach)

    rf_ch4_direct           = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations 
    rf_ch4_indirect           = Variable(index=[time]) #indirect radiative forcing from atmospheric ch4 concentrations 
    rf_ch4_total                   = Variable(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only

end

function interact(M, N)
    d = 1.0 + (M * N)^0.75 * 2.01E-5 + (M * N)^1.52 * M * 5.31E-15
    return 0.47 * log(d)
end

function run_timestep(s::rfch4, t::Int)
    v = s.Variables
    p = s.Parameters

    ch4n2o = interact(p.CH₄_0, p.N₂O_0)
    
    #Direct CH4 radiative forcing (from Myhre, Highwood, & Shine; GRL 2016).
    v.rf_ch4_direct[t] = (0.036 * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0)) - interact(p.CH₄[t], p.N₂O_0) + ch4n2o) * p.scale_CH₄

    #Indirect CH4 forcing from FUND (assume it remains as 40% of old RF equation in FUND, excluding CH4-N2O band overlap). This is what FUND does, not what Marten does in their paper (they include overlaps in indirect stuff). Hector does stratospheric water vapor from ch4 as 5% of direct without overlap band parts of euqation.
    v.rf_ch4_indirect[t] = 0.036 * p.ch4ind * (sqrt(p.CH₄[t]) - sqrt(p.CH₄_0))

    # Calculate total forcing (sum of new direct forcing (shortwave) and FUND's indirect forcing equation)
    v.rf_ch4_total[t] = v.rf_ch4_direct[t] + v.rf_ch4_indirect[t]
end
