using Mimi

@defcomp radiativeforcing begin
    deltat = Parameter()
    rf_co2 = Parameter(index=[time])
    rf_aerosol = Parameter(index=[time])
    forcing_CH₄ = Parameter(index=[time]) #direct radiative forcing from atmospheric ch4 concentrations only
    forcing_O₃ = Parameter(index=[time]) #radiative forcing from tropospheric ozone (direct O3 and ch4-induced ozone production)
    forcing_CH₄_H₂O = Parameter(index=[time]) #indirect CH4 forcing due to production of stratospheric H20 from CH4 oxidation
    rf_other = Parameter(index=[time])
    alpha = Parameter()
    rf = Variable(index=[time])
end

function run_timestep(s::radiativeforcing, t::Int)
    v = s.Variables
    p = s.Parameters

if t==1
    v.rf[t] = 0.0
    else

    v.rf[t] = p.rf_co2[t] + p.forcing_CH₄[t]+ p.forcing_CH₄_H₂O[t] + p.rf_other[t] + (p.alpha * p.rf_aerosol[t]) + p.forcing_O₃[t]
    end    
end
