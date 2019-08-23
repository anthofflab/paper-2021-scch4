module rfco2component
using Mimi

@defcomp rfco2 begin
    atmco2 = Parameter(index=[time])
    rf_co2 = Variable(index=[time])
    F2x_CO₂ = Parameter()
end

function run_timestep(s::rfco2, t::Int)
    v = s.Variables
    p = s.Parameters

    v.rf_co2[t] = p.F2x_CO₂/log(2.0) * log(p.atmco2[t]/p.atmco2[1])
end
end
