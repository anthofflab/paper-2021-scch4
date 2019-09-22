# ----------------------------------------------------------------
# Total CO₂ Emissions (accounting for oxidation of CH₄).
# ----------------------------------------------------------------

@defcomp total_co2_emissions begin

	exogenous_CO₂_emissions = Parameter(index=[time]) # Annual carbon dioxide emissions (GtC yr⁻¹).
	oxidized_CH₄_to_CO₂     = Parameter(index=[time]) # Methane of fossil fule ogirin that has been oxidized to carbon dioxide.

	total_CO₂_emissions     = Variable(index=[time])  # Total CO₂ emissions (GtC yr⁻¹).


    function run_timestep(p, v, d, t)

    	# Calculate total CO₂ emissions as the sum of exogenous sources and CO₂ from oxidized CH₄.
    	v.total_CO₂_emissions[t] = p.exogenous_CO₂_emissions[t] + p.oxidized_CH₄_to_CO₂[t]
    end
end
