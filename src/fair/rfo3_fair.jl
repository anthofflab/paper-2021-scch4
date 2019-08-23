const mol_weight_N = 14.0067  
const mol_weight_NO = 30.01 

@defcomp fair_trop_o3 begin

	model_start_year::Int64 = Parameter() # Different behavior after 1850, so track year here.
	CH₄ = Parameter(index=[time]) # CH4 concentration
	NOx_emissions   = Parameter(index=[time]) # Global NOx emissions in Mt/yr
    CO_emissions    = Parameter(index=[time]) # Global CO emissions in Mt/yr
    NMVOC_emissions = Parameter(index=[time]) # Global non-methane VOC emissions in Mt/yr
	#mol_weight_N = Parameter() #molecular mass of N = 14.0067
	#mol_weight_NO = Parameter() #molecular mass of NO = 30.01
	temperature = Parameter(index=[time])
	CH₄_0 = Parameter() # Initial concentration of CH4 (722 ppb)
	T0 = Parameter()

	forcing_O₃ =  Variable(index=[time])
	F_CH₄= Variable(index=[time])
	F_CO= Variable(index=[time])
	F_NMVOC= Variable(index=[time])
	F_NOx= Variable(index=[time])
end

function temperature_feedback(temperature::Float64)
	if  temperature <= 0.0
    	temperature_feedback = 0.0
    else
    	temperature_feedback = 0.03189267 * exp(-1.34966941 * temperature) - 0.03214807 
    end

    return temperature_feedback
end


function run_timestep(s::fair_trop_o3, t::Int)
    v = s.Variables
    p = s.Parameters

    # Set initial concentration value.
   	#v.CH₄_emiss[t] = p.fossil_emiss[t] + p.natural_emiss[t]

    if (p.model_start_year + t - 1) < 1850
		v.F_CH₄[t] = 0.166/960  * (p.CH₄[t]-p.CH₄_0)
		v.F_CO[t] = 0.058/681.8 * 215.59  * p.CO_emissions[t] / 385.59
		v.F_NMVOC[t] = 0.035/155.84 * 51.97 * p.NMVOC_emissions[t] / 61.97
		v.F_NOx[t] =  0.119/61.16  * 7.31 * (p.NOx_emissions[t] * mol_weight_NO / mol_weight_N) / 11.6
    else
    	v.F_CH₄[t] = 0.166/960  * (p.CH₄[t]-p.CH₄_0)
    	v.F_CO[t] = 0.058/681.8 * (p.CO_emissions[t]-170.0)
		v.F_NMVOC[t] = 0.035/155.84 * (p.NMVOC_emissions[t]-10.0)
		v.F_NOx[t] =  0.119/61.16  * (p.NOx_emissions[t] * mol_weight_NO / mol_weight_N - 4.29)
    end

    #=if p.temperature[t-1] <= 0.0
    	v.temperature_feedback[t] = 0.0
    else
    	v.temperature_feedback[t] = 0.03189267 * exp(-1.34966941 * p.temperature[t-1]) - 0.03214807 
    end
=#
	if t == 1
        # Add option to have non-zero initial condition (base value = 0.0 period 1 temperature anomaly).
    	v.forcing_O₃[t] = v.F_CH₄[t] + v.F_CO[t] + v.F_NMVOC[t] + v.F_NOx[t] + temperature_feedback(p.T0)
    else
    	v.forcing_O₃[t] = v.F_CH₄[t] + v.F_CO[t] + v.F_NMVOC[t] + v.F_NOx[t] + temperature_feedback(p.temperature[t-1])
    end
end
