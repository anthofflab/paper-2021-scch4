#######################################################################################################################
# CALCULATE MARGINAL DAMAGES FROM A CH₄ EMISSION PULSE WITH FUND.
#######################################################################################################################
# Description: This function uses the FUND model to calculate marginal climate damages that occur from a CH₄ emissions
#              pulse based on the SNEASY+CH4 climate projections.
#
# Function Arguments:
#
#       base_temp  = An array of baseline SNEASY+CH4 temperature projections (each row = climate projection using a new
#                    posterior parameter sample, each column = a new year).
#       base_co2    = An array of baseline SNEASY+CH4 CO₂ concentration projections (each row = climate projection using
#                     a new posterior parameter sample, each column = a new year).
#       pulse_temp = An array of SNEASY+CH4 temperature projections similar to `base_temp`, but with a pulse of CH₄
#                    emissions in a user-specified year.
#       pulse_co2  = An array of SNEASY+CH4 CO₂ concentration projections similar to `base_co2`, but with a pulse of CH₄
#                    emissions in a user-specified year.
#       end_year   = The final year to run DICE.
#----------------------------------------------------------------------------------------------------------------------

function fund_damages(base_temp::Array{Float64,2}, base_co2::Array{Float64,2}, pulse_temp::Array{Float64,2}, pulse_co2::Array{Float64,2}, end_year::Int)

    #Create vector of years FUND is run (note FUND initializes in 1950).
    fund_years = collect(1950:end_year)

    #Find indices in SNEASY+CH4 climate projections that correspond to FUND years (assuming SNEASY+CH4 initialized in 1765).
    sneasy2fund_indices = findall((in)(fund_years), 1765:end_year)

    #Extract climate output that corresponds to FUND years (each row is a new model run, each column a new year).
    base_temp  = base_temp[:, sneasy2fund_indices]
    base_co2   = base_co2[:, sneasy2fund_indices]
    pulse_temp = pulse_temp[:, sneasy2fund_indices]
    pulse_co2  = pulse_co2[:, sneasy2fund_indices]

    #Caluclate number of times to run FUND (equal to number of posterior parameter samples used in climate projections).
    number_samples = size(base_temp, 1)

    #Get a base and CH₄ emissions pulse version of FUND model.
    fund_base  = create_iam_fund(end_year=end_year)
    fund_pulse = create_iam_fund(end_year=end_year)

    # Pre-allocate arrays to store relevant FUND output (Note: FUND has 16 geographic regions).
    marginal_damages             = zeros(length(fund_years), 16, number_samples)
    regional_population_base     = zeros(length(fund_years), 16, number_samples)
    regional_consumption_base    = zeros(length(fund_years), 16, number_samples)

    # Caluclate marginal damages and per capita consumption values for each posterior parameter climate projection.
    for i in 1:number_samples

        # Wrap code in a try/catch statement in case extreme climate values cause a model error.
        try

            # FUND cannot accept negative temperature values (which may occur in a few early periods due to superimposed noise). In both IAMs, set negative anomolies to 1e-15 to maintain consistency.
            if minimum(base_temp[i, :]) < 0.0
                base_temp[i, findall(x -> x <= 0.0, base_temp[i,:])] .= 1e-15
                pulse_temp[i, findall(x -> x <= 0.0, pulse_temp[i,:])] .= 1e-15
            end

            # Assign SNEASY+CH4 projections for temperature and CO₂ concentrations to relevant FUND components.
            update_param!(fund_base, :temp, base_temp[i, :])
            update_param!(fund_base, :acco2, base_co2[i, :])

            update_param!(fund_pulse, :temp, pulse_temp[i, :])
            update_param!(fund_pulse, :acco2, pulse_co2[i, :])

            #Run the two versions of FUND.
            run(fund_base)
            run(fund_pulse)

            #Calculate annual marginal climate damages (note: 1950 damage = `missing`, so set to 0.0 for convenience).
            marginal_damages[2:end,:,i] = fund_pulse[:impactaggregation, :loss][2:end,:] .- fund_base[:impactaggregation, :loss][2:end,:]

            # Calculate population for baseline scenario (note: FUND population units = millions of people).
            regional_population_base[:,:,i] = fund_base[:socioeconomic, :population] .* 1e6

            # Calculate regional consumption.
            regional_consumption_base[:,:,i] = fund_base[:socioeconomic, :consumption]

        # If climate projections cause a model error, set FUND output to -99999.99.
        catch
            marginal_damages[:,:,i] .= -99999.99
            regional_population_base[:,:,i] .= -99999.99
            regional_consumption_base[:,:,i] .= -99999.99
        end
    end

    # Check if any samples caused a model error.
    error_indices = findall(x-> x == -99999.99, marginal_damages[1,1,:])
    good_indices  = findall(x-> x != -99999.99, marginal_damages[1,1,:])

    # Return results.
    return marginal_damages[:,:,good_indices], regional_population_base[:,:,good_indices], regional_consumption_base[:,:,good_indices], error_indices, good_indices
end



#######################################################################################################################
# CALCULATE THE SC-CH4 USING FUND.
#######################################################################################################################
# Description: This function uses the marginal climate damage estimates from DICE to calculate the SC-CH4.
#
# Function Arguments:
#
#       marginal_damages     = Annual marginal climate damages (row = year, column = FUND region, 3rd dimension = model run
#                              using a new posterior climate parameter sample).
#       regional_consumption = Regional consumption levels.
#       regional_population  = Regional population levels.
#       pulse_year           = The year that the CH₄ emission pulse is added.
#       end_year             = The final year to run FUND.
#       constant             = Boolean parameter for whether or not to use a constant discount rate (true = use constant discount rate).
#       η                    = Parameter representing the degree of intertemporal inequality aversion.
#       γ                    = Parameter representing the degree of regional inequality aversion.
#       ρ                    = Pure rate of time preference.
#       dollar_conversion    = Inflation conversion factor to scale dollar units to a different year.
#       equity_weighting     = Should the model produce equity-weighted SC-CH4 estimates? (true = produce equity-weighted estimates).
#----------------------------------------------------------------------------------------------------------------------

function fund_scch4(marginal_damages::Array{Float64,3}, regional_consumption::Array{Float64,3}, regional_population::Array{Float64,3}, pulse_year::Int, end_year::Int; constant::Bool=true, η::Float64=1.5, γ::Float64=1.5, ρ::Float64=0.01, dollar_conversion::Float64=1.35, equity_weighting=false)

    #Create vector of years FUND is run (note FUND initializes in 1950).
    fund_years = collect(1950:end_year)

    # Find timestep index when CH₄ emission pulse occurs.
    pulse_index = findall(x -> x == pulse_year, fund_years)[1]

    # Calculate number of SC-CH4 values to calculate.
    number_samples = size(marginal_damages, 3)

    # Calculate global and regional per capita consumption values (for discounting & equity-weighting).
    pc_consumption_regional = regional_consumption ./ regional_population
    pc_consumption_global   = sum(regional_consumption, dims=2) ./ sum(regional_population, dims=2)

    # Allocate an array to hold either constant or Ramsey discount factors.
    df = zeros(length(fund_years))

    # If using a constant discount rate, calculate constant discount factors for annual timesteps.
    if constant == true
        calculate_discount_factors!(df, ones(length(fund_years)), ρ, 0.0, fund_years, pulse_year, pulse_index)
    end

    #-----------------------------------------------------------------------
    # Loop through each marginal damage estimate and calculate the SC-CH4.
    #-----------------------------------------------------------------------

    # Calculate the SC-CH4 for each posterior parameter sample without equity-weighting.
    if equity_weighting == false

        # Pre-allocate array to store results and SC-CH4 estimates.
        discounted_global_marginal_damages = zeros(number_samples, length(fund_years))
        scch4 = zeros(number_samples)

        # Loop through damages resulting from each posterior parameter sample and calculate the SC-CH4.
        for i = 1:number_samples
            # If not using a constant discount rate, calculate a Ramsey discount rate for each posterior parameter sample based on global per capita consumption levels.
            if constant == false
                calculate_discount_factors!(df, pc_consumption_global[:,1,i], ρ, η, fund_years, pulse_year, pulse_index)
            end

            # Calculate discounted global annual marginal damages (summed across the FUND regions).
            discounted_global_marginal_damages[i,:] = @views sum(marginal_damages[:,:,i] .* df, dims=2)

            # Calculate the SC-CH4 as the sum of discounted annual marginal damages (scaled by inflation conversion factor for desired year).
            scch4[i] = @views sum(discounted_global_marginal_damages[i,:]) * dollar_conversion
        end

        # Return SC-CH4 and discounted annual marginal damage values.
        return scch4, discounted_global_marginal_damages

    # Calculate the SC-CH4 for each posterior parameter sample using equity-weighting.
    elseif equity_weighting == true

        # Pre-allocate array to store equity-weighted SC-CH4 estimates for each region as well as some temporary variables (for convenience).
        scch4           = zeros(number_samples, 16)
        tempvar_1       = zeros(16)
        annual_vals     = zeros(length(fund_years))
        cde_consumption = zeros(length(fund_years), number_samples)

        # Following the approach of Anthoff & Emmerling (2019), calculate equity-weighted SC-CH4 values for each normalization region.
        for i = 1:number_samples
            for r = 1:16
                for t = pulse_index:length(fund_years)
                    cde_consumption[t,i] = sum(regional_population[t,:,i] ./ sum(regional_population[t,:,i]) .* pc_consumption_regional[t,:,i] .^(1-γ)) ^ (1/(1-γ))
                    tempvar_1[:] .= @views (pc_consumption_regional[t,:,i] ./ pc_consumption_regional[pulse_index,r,i]) .^ -γ
                    tempvar_2    = (cde_consumption[t,i] ./ cde_consumption[pulse_index,i]) .^ (γ-η)
                    annual_vals[t] = @views sum(tempvar_1 .* tempvar_2 .* (1+ρ) ^ (-t+pulse_index) .* marginal_damages[t,:,i])
                end
                scch4[i,r] = sum(annual_vals) * dollar_conversion
            end
        end

        # Return equity-weighted SC-CH4 values.
        return scch4
    end
end
