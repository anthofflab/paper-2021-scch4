include("../create_models/create_iam_dice.jl")

using Interpolations


#function dice_damages(base_temp::Array{Union{Missing,Float64},2}, pulse_temp::Array{Union{Missing,Float64},2}, end_year::Int)
function dice_damages(base_temp::Array{Float64,2}, pulse_temp::Array{Float64,2}, end_year::Int)


           #Create vector of years DICE is run (note DICE2013 has 5 year timesteps)
           dice_years = collect(2010:5:end_year)

           #Find indices in climate results that correspond to DICE years (assuming SNEASY+CH4 initialized in 1765).
           sneasy2dice_indices = findall((in)(dice_years), 1765:end_year)

           #Extract climate output that corresponds to every 5th year for DICE timesteps (each row is a new model run, each column a new year).
           base_temp  = base_temp[:, sneasy2dice_indices]
           pulse_temp = pulse_temp[:, sneasy2dice_indices]

           #Caluclate number of times to run DICE (equal to number of posterior parameter samples used).
           number_samples = size(base_temp, 1)

           # Create a vector of annual years for DICE time horizon (with DICE initializing in 2010).
           annual_years = collect(2010:end_year)

           #Get two versions (base and marginal) of DICE model
           dice_base  = create_iam_dice(end_year=end_year)
           dice_pulse = create_iam_dice(end_year=end_year)

           # Pre-allocate arrays to store DICE output.
           damages_base  = zeros(length(dice_years))
           damages_pulse = zeros(length(dice_years))
           annual_marginal_damages = zeros(number_samples, length(annual_years))
           pc_consumption_base = zeros(number_samples, length(annual_years))


           # Caluclate marginal damages and per capita consumption values for each posterior parameter sample.
           for i in 1:number_samples

            # Wrap code in a try/catch statement in case extreme ECS value causes a model error.
            try

               # FUND cannot accept negative temperature values (which may occur in a few early periods due to superimposed noise). In both IAMs, set negative anomolies to 0.0 to maintain consistency.
                if minimum(base_temp[i, :]) < 0.0
                    base_temp[i, findall(x -> x <= 0.0, base_temp[i,:])] .= 1e-15
                    pulse_temp[i, findall(x -> x <= 0.0, pulse_temp[i,:])] .= 1e-15
                end

               # Assign SNEASY+CH4 projections to DICE damage module.
               update_param!(dice_base,  :TATM, base_temp[i, :])
               update_param!(dice_pulse, :TATM, pulse_temp[i, :])

               #Run the two versions of DICE.
               run(dice_base)
               run(dice_pulse)

               #Calculate damages and convert from trillions to dollars
               damages_base[:]  = dice_base[:damages, :DAMAGES] .* 1e12
               damages_pulse[:] = dice_pulse[:damages, :DAMAGES] .* 1e12

               annual_marginal_damages[i,:] = dice_interpolate((damages_pulse .- damages_base), 5) #./ 10^6
               pc_consumption_base[i,:] = dice_interpolate(dice_base[:neteconomy, :CPC], 5)

              catch
                annual_marginal_damages[i,:] .= -99999.99
                pc_consumption_base[i,:] .= -99999.99
              end
           end

            # Distinguish between model indices that caused a model error or not.
            error_indices = findall(x-> x == -99999.99, annual_marginal_damages[:,1])
            good_indices  = findall(x-> x != -99999.99, annual_marginal_damages[:,1])

           return annual_marginal_damages, pc_consumption_base, error_indices, good_indices
       end



function dice_scch4(annual_marginal_damages::Array{Float64,2}, pc_consumption::Array{Float64,2}, pulse_year::Int, end_year::Int; constant::Bool, η::Float64, ρ::Float64, dollar_conversion::Float64)

    #Create vector of years DICE is run (note DICE2013 has 5 year timesteps)
    dice_annual_years = collect(2010:end_year)

    # Find index in annual years when CH₄ emission pulse occurs.
    pulse_index_annual = findall(x -> x == pulse_year, dice_annual_years)[1]

    # Calculate number of SC-CH4 values to calculate.
    number_samples = size(annual_marginal_damages, 1)

    #Pre-allocate arrays to store model results.
    discounted_marginal_annual_damages = zeros(number_samples, length(dice_annual_years))
    scch4 = zeros(number_samples)

    # Allocate an array to hold either constant or Ramsey discount factors.
    df = zeros(length(dice_annual_years))

    # Calculate a constant discount rate for annual timesteps.
    if constant == true
        calculate_discount_factors!(df, ones(length(dice_annual_years)), ρ, 0.0, dice_annual_years, pulse_year, pulse_index_annual)
    end

    #---------------------------------------------------------------------
    # Loop through each marginal damage estimate and calculate the SC-CH4.
    #---------------------------------------------------------------------
    for i = 1:number_samples

        # If not using a constant discount rate, calculate a Ramsey discount rate for each posterior parameter sample.
        if constant == false
            calculate_discount_factors!(df, pc_consumption[i,:], ρ, η, dice_annual_years, pulse_year, pulse_index_annual)
        end

        # Calculate discounted annual marginal damages.
        discounted_marginal_annual_damages[i,:] = annual_marginal_damages[i,:] .* df

        # Calculate the SC-CH4 as the sum of discounted annual marginal damages (scaled by inflation conversion factor for desired year).
        scch4[i] = sum(discounted_marginal_annual_damages[i,:]) * dollar_conversion
    end

    return scch4, discounted_marginal_annual_damages
end
