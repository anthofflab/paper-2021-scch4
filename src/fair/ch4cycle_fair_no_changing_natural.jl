# Conversion between ppb/ppt concentrations and Mt/kt emissions
    # Weight of atmosphere in kilograms = 5.1252 x 10e18
    # From FAIR1.3 code notes: "in the RCP databases ppb = Mt and ppt = kt so factor always 1e18"
    # Molecular mass of CH₄ = 16.04
    # Molecular mass of air = 28.97
const mol_weight_CH₄ = 16.04
const mol_weight_CO₂ = 44.01
const emiss_to_ppb =  (5.1352e18 / 1e18) * (mol_weight_CH₄ / 28.97)

@defcomp ch4cycle_fair begin

    model_start_year::Int64 = Parameter() # Constant natural emissions after 2005, so track year here.
    constant_natural_CH₄ = Parameter() # Constant value for natural CH4 emissions after 2005 (default = 190.5807)
    fossil_emiss = Parameter(index=[time])
    #natural_emiss = Parameter(index=[time])
    CH₄_0 = Parameter() # Initial concentration of CH4 (722 ppb)
    τ   = Parameter() # Atmospheric (e-folding) lifetime of CH₄ (dfault = 9.3 years)
    fossil_frac = Parameter(index=[time]) # Fraciton of anthropogenic CH4 attributable to fossil sources (time-varying from RCP scenarios...taken from original FAIR data).
    oxidation_frac = Parameter() # Fraction of methane lost trhough reaction with hydroxyl radical that is converted to CO₂ (default = 0.61)
    #mol_weight_CH₄ = Parameter() #molecular mass of CH4 = 16.04
    #mol_weight_CO₂ = Parameter() #molecular mass of CO2 = 44.01
    #CH₄_emiss = Variable(index=[time])
    CH₄ = Variable(index=[time])
    oxidised_CH₄_to_CO₂ = Variable(index=[time]) # CH4 oxidation to CO2
end

function run_timestep(s::ch4cycle_fair, t::Int)
    v = s.Variables
    p = s.Parameters

    # Set initial concentration value.
    #v.CH₄_emiss[t] = p.fossil_emiss[t] + p.natural_emiss[t]

    if t == 1
        v.CH₄[t] = p.CH₄_0
        v.oxidised_CH₄_to_CO₂[t] = 0.0
    else
        #### REMOVE TIME VARYING NATURAL EMISSIONS

        # FAIR 1.3 assumes natural CH₄ emissions are constant after 2005, allow this paramter to be uncertain.
        #if (p.model_start_year + t - 1) <= 2005
            # Create temporary emission variables (to make code easier to read). Note natural emissions always for period t in FAIR code.
        #    emiss_prev = p.fossil_emiss[t-1] + p.natural_emiss[t]
        #    emiss_curr = p.fossil_emiss[t] + p.natural_emiss[t] 
        #else
            # If after 2005, set user defined constant natural emissions rate for CH₄.
            emiss_prev = p.fossil_emiss[t-1] + p.constant_natural_CH₄
            emiss_curr = p.fossil_emiss[t] + p.constant_natural_CH₄ 
        
        v.CH₄[t] = v.CH₄[t-1] - v.CH₄[t-1] * (1.0 - exp(-1/p.τ)) + 0.5 * (emiss_prev + emiss_curr) * (1.0/emiss_to_ppb)
        
        # Calculate CO2 from oxidized CH4 (bound it below at 0.0).
        v.oxidised_CH₄_to_CO₂[t] = max(((v.CH₄[t-1]-p.CH₄_0) * (1.0 - exp(-1.0/p.τ)) * (mol_weight_CO₂/mol_weight_CH₄ * 0.001 * p.oxidation_frac * p.fossil_frac[t])), 0.0)
    end
end
