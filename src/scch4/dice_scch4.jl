# Load file to run version of DICE that can incorporate SNEASY+CH4 results.
include("../create_models/create_iam_dice.jl")

#######################################################################################################################
# CALCULATE MARGINAL DAMAGES FROM A CH₄ EMISSION PULSE WITH DICE.
#######################################################################################################################
# Description: This function uses the DICE model to calculate marginal climate damages that occur from a CH₄ emissions
#              pulse based on the SNEASY+CH4 climate projections.
#
# Function Arguments:
#
#       base_temp  = An array of baseline SNEASY+CH4 temperature projections (each row = climate projection using a new 
#                    posterior parameter sample, each column = a new year).
#       pulse_temp = An array of SNEASY+CH4 temperature projections similar to `base_temp`, but with a pulse of CH₄
#                    emissions in a user-specified year.
#       end_year   = The final year to run DICE.
#----------------------------------------------------------------------------------------------------------------------

function dice_damages(base_temp::Array{Float64,2}, pulse_temp::Array{Float64,2}, end_year::Int)

    #Create vector of years to run DICE for (note DICE2013 has 5 year timesteps).
    dice_years = collect(2010:5:end_year)

    #Find indices in SNEASY+CH4 climate projections that correspond to DICE years (assuming SNEASY+CH4 initialized in 1765).
    sneasy2dice_indices = findall((in)(dice_years), 1765:end_year)

    #Extract climate output that corresponds to every 5th year for DICE timesteps (each row is a new model run, each column a new year).
    base_temp  = base_temp[:, sneasy2dice_indices]
    pulse_temp = pulse_temp[:, sneasy2dice_indices]

    #Caluclate number of times to run DICE (equal to number of posterior parameter samples used in climate projections).
    number_samples = size(base_temp, 1)

    # Create a vector of annual years for DICE time horizon (with DICE initializing in 2010).
    annual_years = collect(2010:end_year)

    #Get a base and CH₄ emissions pulse version of DICE model.
    dice_base  = create_iam_dice(end_year=end_year)
    dice_pulse = create_iam_dice(end_year=end_year)

    # Pre-allocate arrays to store relevant DICE output.
    damages_base  = zeros(length(dice_years))
    damages_pulse = zeros(length(dice_years))
    annual_marginal_damages = zeros(number_samples, length(annual_years))
    pc_consumption_base = zeros(number_samples, length(annual_years))

    # Caluclate marginal damages and per capita consumption values for each posterior parameter climate projection.
    for i in 1:number_samples

        # Wrap code in a try/catch statement in case extreme climate values cause a model error.
        try

            # FUND cannot accept negative temperature values (which may occur in a few early periods due to superimposed noise). In both IAMs, set negative anomolies to 1e-15 to maintain consistency.
            if minimum(base_temp[i, :]) < 0.0
                base_temp[i, findall(x -> x <= 0.0, base_temp[i,:])] .= 1e-15
                pulse_temp[i, findall(x -> x <= 0.0, pulse_temp[i,:])] .= 1e-15
            end

            # Assign SNEASY+CH4 temperature projections to DICE climate damage module.
            update_param!(dice_base,  :TATM, base_temp[i, :])
            update_param!(dice_pulse, :TATM, pulse_temp[i, :])

            #Run the two versions of DICE.
            run(dice_base)
            run(dice_pulse)

            #Calculate damages and convert from $trillions to dollars.
            damages_base[:]  = dice_base[:damages, :DAMAGES] .* 1e12
            damages_pulse[:] = dice_pulse[:damages, :DAMAGES] .* 1e12

            # Calculate marginal damages and interpolate to annual values.
            annual_marginal_damages[i,:] = dice_interpolate((damages_pulse .- damages_base), 5)

            # Convert baseline per capita consumption levels to annual values.
            pc_consumption_base[i,:] = dice_interpolate(dice_base[:neteconomy, :CPC], 5)

        # If climate projections cause a model error, set DICE output to -99999.99.
        catch
            annual_marginal_damages[i,:] .= -99999.99
            pc_consumption_base[i,:] .= -99999.99
        end
    end

    # Distinguish between DICE model indices that caused a model error and those that did not.
    error_indices = findall(x-> x == -99999.99, annual_marginal_damages[:,1])
    good_indices  = findall(x-> x != -99999.99, annual_marginal_damages[:,1])

    # Return results.
    return annual_marginal_damages, pc_consumption_base, error_indices, good_indices
end



#######################################################################################################################
# CALCULATE THE SC-CH4 USING DICE.
#######################################################################################################################
# Description: This function uses the marginal climate damage estimates from DICE to calculate the SC-CH4.
#
# Function Arguments:
#
#       annual_marginal_damages = Anual marginal climate damages (interpolated to annual values).
#       pc_consumption          = Per capita consumption (interpolated to annual values).
#       pulse_year              = The year that the CH₄ emission pulse is added.
#       end_year                = The final year to run DICE.
#       constant                = Boolean parameter for whether or not to use a constant discount rate (true = use constant discount rate).
#       η                       = Elasticity of marginal utility of consumption.
#       ρ                       = Pure rate of time preference.
#       dollar_conversion       = Inflation conversion factor to scale dollar units to a different year.
#----------------------------------------------------------------------------------------------------------------------

function dice_scch4(annual_marginal_damages::Array{Float64,2}, pc_consumption::Array{Float64,2}, pulse_year::Int, end_year::Int; constant::Bool, η::Float64, ρ::Float64, dollar_conversion::Float64)

    #Create vector of years DICE is run (note DICE2013 has 5 year timesteps).
    dice_annual_years = collect(2010:end_year)

    # Find timestep index in annual years when CH₄ emission pulse occurs.
    pulse_index_annual = findall(x -> x == pulse_year, dice_annual_years)[1]

    # Calculate number of SC-CH4 values to calculate.
    number_samples = size(annual_marginal_damages, 1)

    #Pre-allocate arrays to store model results.
    discounted_marginal_annual_damages = zeros(number_samples, length(dice_annual_years))
    scch4 = zeros(number_samples)

    # Allocate an array to hold either constant or Ramsey discount factors.
    df = zeros(length(dice_annual_years))

    # If using a constant discount rate, calculate constant discount factors for annual timesteps.
    if constant == true
        calculate_discount_factors!(df, ones(length(dice_annual_years)), ρ, 0.0, dice_annual_years, pulse_year, pulse_index_annual)
    end

    #-----------------------------------------------------------------------
    # Loop through each marginal damage estimate and calculate the SC-CH4.
    #-----------------------------------------------------------------------
    for i = 1:number_samples

        # If not using a constant discount rate, calculate a Ramsey discount factor for each posterior parameter sample.
        if constant == false
            calculate_discount_factors!(df, pc_consumption[i,:], ρ, η, dice_annual_years, pulse_year, pulse_index_annual)
        end

        # Calculate discounted annual marginal climate damages.
        discounted_marginal_annual_damages[i,:] = annual_marginal_damages[i,:] .* df

        # Calculate the SC-CH4 as the sum of discounted annual marginal damages (scaled by inflation conversion factor for desired year).
        scch4[i] = sum(discounted_marginal_annual_damages[i,:]) * dollar_conversion
    end

    return scch4, discounted_marginal_annual_damages
end
