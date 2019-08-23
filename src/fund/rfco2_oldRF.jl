module rfco2component
using Mimi

@defcomp rfco2 begin
    atmco2 = Parameter(index=[time])
    rf_co2 = Variable(index=[time])
end

function run_timestep(s::rfco2, t::Int)
    v = s.Variables
    p = s.Parameters

    v.rf_co2[t] = 3.7 * log(p.atmco2[t]/p.atmco2[1])/log(2.0)
end

end