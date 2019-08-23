#Various functions used for Sneasy Methane work.

# Function to calculate scaling coefficient so CO2 RF is consistent with user
# supplied F2x parameter (default = 3.7 for old, 3.801 for Etiman update)

function co2_rf_scale(F2x::Float64, CO₂_0::Float64, N₂O_0::Float64)

    # Calcualte forcing from doubling of CO2 (following apporach from FAIR 1.3)
    co2_doubling = (-2.4e-7 * CO₂_0^2 + 7.2e-4 * CO₂_0 - 2.1e-4 * N₂O_0 + 5.36) * log(2)

    # Calculate scaling factor, given user supplied F2x parameter.
    CO₂_scale = F2x / co2_doubling

    return CO₂_scale
end


# FUNCTION TO BACK OUT NATURAL EMISSIONS FOR FAIR 1.3 (currently don't use)
#CH4 = RCP concnetrations
#fossil_emiss = RCP CH4 emissions

function nat_emissions(CH₄, fossil_emiss, lifetime, start_year, end_year)

    n_vals = length(start_year:end_year)
    natural_emissions = zeros(n_vals)
    index_2005 = findin(collect(start_year:end_year), 2005)[1]

    mol_weight_CH₄ = 16.04
    emiss_to_ppb =  (5.1352e18 / 1e18) * (mol_weight_CH₄ / 28.97)

    for t = 1:index_2005
        b = 0.5 * (1.0 / emiss_to_ppb)

        if t == 1
            a = CH₄[t] - (CH₄[t] - CH₄[t]*(1-exp(-1.0/lifetime)))
            c = fossil_emiss[t] + fossil_emiss[t]
        else
            a = CH₄[t] - (CH₄[t-1] - CH₄[t-1]*(1-exp(-1.0/lifetime)))
            c = fossil_emiss[t] + fossil_emiss[t-1]
        end

        natural_emissions[t] = ((a/b)-c) / 2.0
    end

    natural_emissions[(index_2005+1):end] = natural_emissions[index_2005]

    return natural_emissions
end

###########################################################################################

#Acceptance Rate for MCMC results

function mcmc_acceptance(v)
    accepted = 1
    n = length(v)

    for i in 2:n
        if v[i] != v[i-1]
            accepted += 1
        end
    end

    return accepted/n
end

###########################################################################################

# Simulate stationary AR(1) process

function ar1_sim(N, rho, sigma)
    x = zeros(N)
    #x[1] = sigma/sqrt(1-rho^2)
    x[1] = rand(Normal(0, (sigma[1]/sqrt(1-rho^2))))
    for i in 2:N
        x[i] = rho * x[i-1] + rand(Normal(0, sigma))
    end
    return x
end

###########################################################################################

# Simulate stationary AR(1) process with time varying observation error.

function ar1_hetero_sim(N, rho, sigma)
    x = zeros(N)
    #June 18 fixed this to have initial value vary
    #x[1] = sigma[1]/sqrt(1-rho^2)
    x[1] = rand(Normal(0, (sigma[1]/sqrt(1-rho^2))))
    for i in 2:N
        x[i] = rho * x[i-1] + rand(Normal(0, sigma[i]))
    end
    return x
end

###########################################################################################
# Function to replicate observation errors for periods without observations

# This only works if model starts in 1850
    function replicate_errors(start_year, end_year, error_data)

        model_years = collect(start_year:end_year)
        #Initialize new vector of errors (assume )
        errors = zeros(length(model_years))
        # Find indices for periods that have observation errors.
        err_indices = find(x-> !isna(x), error_data)
        #Replicate 1st error for all periods prior to start of observations.
        errors[1:err_indices[1]] = mean(error_data[err_indices[1]:err_indices[10]])
        #Add all errors for periods with observations.
        errors[err_indices[2]:err_indices[end]] = error_data[err_indices[2]:err_indices[end]]
        # Replicate last error for all periods after observation data.
        errors[(err_indices[end]+1):end] = mean(error_data[err_indices[end-9]:err_indices[end]])

        return errors
    end

###########################################################################################
# Function to do a single AR1 process for pooled statistical parameters (ice core (noisy) to air measurements (not noisy))
# TODO: Run this by Klaus
#=
function ch4_mixed_noise(N1, rho1, sigma1, N2, rho2, sigma2)
    x = zeros(N1+N2)
    x[1] = sigma1/sqrt(1-rho1^2)
    for i in 2:N1
        x[i] = rho1 * x[i-1] + rand(Normal(0, sigma1))
    end

    for i in (N1+1):(N1+N2)
        x[i] = rho2 * x[i-1] + rand(Normal(0, sigma2))
    end
    return x
end
=#
# Note CH4 ice has constant measurment error, inst has time varying measurement error so need to index appropriately
function ch4_mixed_noise(start_year, end_year, ρ_ice, σ_ice, err_ice, ρ_inst, σ_inst, err_inst)
    noise_vector = zeros(length(start_year:end_year))
    # Start year to 1983 is AR1 process using ice core statistical parameters.
    n_years = length(start_year:end_year)
    n_ice = length(start_year:1983)
    n_inst = length(1984:end_year)

    # Noise for period covered by ice core data
    noise_vector[1] = σ_ice/sqrt(1-ρ_ice^2)
    for t = 2:n_ice
        noise_vector[t] = ρ_ice * noise_vector[t-1] + rand(Normal(0, sqrt(σ_ice^2 + err_ice^2)))
    end
    # Noise for period covered by instrumental data
    for t = (n_ice+1):n_years
        noise_vector[t] = ρ_inst * noise_vector[t-1] + rand(Normal(0, sqrt(σ_inst^2 + err_inst[t]^2)))
    end
    return noise_vector
end



###########################################################################################
function co2_mixed_noise(start_year, end_year, σ_ice, σ_inst, err_ice, err_inst, ρ_inst)
    # Allocate initial vector.
    noise_vector = zeros(length(start_year:end_year))
    # Start year to 1958 is a iid noise process for ice core data.
    n_years = length(start_year:end_year)
    n_ice = length(start_year:1958)
    n_inst = length(1959:end_year)

    for t = 1:n_ice
        noise_vector[t] = rand(Normal(0.0, sqrt(σ_ice^2 + err_ice^2)))
    end

    for t = (n_ice+1):n_years
        noise_vector[t] = ρ_inst * noise_vector[t-1] + rand(Normal(0.0, sqrt(σ_inst^2 + err_inst^2)))
    end
    return noise_vector
end

###################################################################################
#################################################################################################################
#----------------------------------------------------------------------------------------------------------------
# LINEARLY INTERPOLATE DICE RESULTS INTO ANNUAL TIMESTEPS
#
# Function that uses linear interpolation to create annual time series values from DICE results. This is hardcoded
# to work for the standard DICE time period (2010-2305) with 5 year timesteps.
#
# Function argument descriptions
#   data        = The DICE results to be interpolated.
#   start_year  = First year in the timeseries to be interpolated.
#   end_year    = Last year in the timeseries to be interpolated.
#----------------------------------------------------------------------------------------------------------------
#=
function dice_interpolate(data, start_year, end_year, marginal_year)
    results = zeros(length(start_year:end_year))
    # DICE marginal year index (for annual years)
    annual_marg_index = findin(collect(start_year:end_year), marginal_year)[1]
    # Marginal index based on DICE 5 year time steps
    dice_marg_index = findin(collect(start_year:5:end_year), marginal_year)[1]

    indices_with_data = collect(annual_marg_index:5:length(start_year:end_year))

    #Set every 5th index to the DICE results
    results[indices_with_data] = data[dice_marg_index:end]

    # For blank spaces, fill in with a linear interpolation between the two endpoints.
    for i in dice_marg_index:length(indices_with_data)
        if i < length(indices_with_data)
        diff = (results[indices_with_data[i+1]] - results[indices_with_data[i]]) / 5.0
            results[(indices_with_data[i]+1)] = results[indices_with_data[i]] + (1 * diff)
            results[(indices_with_data[i]+2)] = results[indices_with_data[i]] + (2 * diff)
            results[(indices_with_data[i]+3)] = results[indices_with_data[i]] + (3 * diff)
            results[(indices_with_data[i]+4)] = results[indices_with_data[i]] + (4 * diff)
        end
    end
    return results
end
=#

function dice_interpolate(data, spacing)

    # Create an interpolation object for the data (assume first and last points are end points, e.g. no interpolation beyond support).
    interp_linear = interpolate(data, BSpline(Linear()), OnGrid())

    # Create points to interpolate for (based on spacing term). DICE has 10 year time steps.
    interp_points = collect(1:(1/spacing):length(data))

    # Carry out interpolation.
    return interp_linear[interp_points]
end

####################################################################################
#Function to calculate confidence intervals from mcmc runs

#Have data in form: Column 1 = Year, Column 2 = Chain Means, Columns 3+ = Bootstraps

#Gives back data in form Year, Chain Mean, Upper1, Lower1, Upper2, Lower2, Confidence for Plotting

function confidence_int(years, my_data, conf_1_percent, conf_2_percent)
    alpha1 = 1-conf_1_percent
    alpha2 = 1-conf_2_percent
    n_years = size(my_data, 1)

    confidence = DataFrame(Year=years[:,1], Mean_Chain=zeros(n_years), Lower1=zeros(n_years), Upper1=zeros(n_years), Lower2=zeros(n_years), Upper2=zeros(n_years))

    my_data = convert(Array, my_data)

    my_data[find(isnan(my_data))] = - 9999.99

    confidence[:,2]=vec(mapslices(mean, my_data, 2))

    for i in 1:n_years
        confidence[:Lower1][i] = quantile(vec(my_data[i,:]), alpha1 /2)
        confidence[:Upper1][i] = quantile(vec(my_data[i,:]), 1-alpha1 /2)
        confidence[:Lower2][i] = quantile(vec(my_data[i,:]), alpha2/2)
        confidence[:Upper2][i] = quantile(vec(my_data[i,:]), 1-alpha2/2)
    end

    rename!(confidence, :Lower1, Symbol(join(["LowerConf" conf_1_percent], '_')))
    rename!(confidence, :Upper1, Symbol(join(["UpperConf" conf_1_percent], '_')))
    rename!(confidence, :Lower2, Symbol(join(["LowerConf" conf_2_percent], '_')))
    rename!(confidence, :Upper2, Symbol(join(["UpperConf" conf_2_percent], '_')))

    return(confidence)
end

###########################################################################################



# Function to calculate distribution of climate sensitivity (applied to SCM estimates in Marten and Newbold (2011) 
#   working paper, SCC estimates in Marten (2011), and derived in Roe and Baker (2007).

# Uncertinaty in climate feedbacks is represented by a normal distribution. It is parameterized with mean = 0.61
#   and standard deviation = 0.17, resulting in a median climate sensitivity of approximately 3C and a 66% confidence
#   interval of 2C to 4.5C.
#
# The function argument 'n' specifies how many sample draws to take from the distribution.
#
# The distriubtion is truncated to have a range from 0-10 degrees C

function clim_sens_distribution(n::Int64)
    clim_sens = zeros(n)

    #Set climate feedback distribution parameters (from Marten & Newbold 2012, Energy Policy)
    f_mean = 0.6198
    f_std_dev = 0.1841

    #Find climate sensitivity values, truncated distribution at 0 and 10.
    x=0
    for i in 1:n
        while x > 10.0 || x <= 0.0
            x = 1.2 / (1 - rand(Normal(f_mean, f_std_dev)))
            end
        clim_sens[i] = x
        x=0
    end

    return(clim_sens)
end

##############################################################################################
# TODO: Add description of what this does with function argument descriptions

function run_mcmc_all(mcmc_file::String, n_steps::Int64, burn_in::Int64, save_results::Bool, save_full_chain::Bool, fund::Bool, hector::Bool, magicc::Bool)

    if fund
        println("Beginning MCMC calibration of SneasyCH4-F.")
        include("../calibration/sneasy_fund/climassim.jl")
        run_mcmc_f(mcmc_file, n_steps, burn_in, save_results, save_full_chain)
        println()
        println("Completed SneasyCH4-F MCMC calibration.")
    end

    if hector
        println("Beginning MCMC calibration of SneasyCH4-H.")
        include("../calibration/sneasy_hector/climassim.jl")
        run_mcmc_h(mcmc_file, n_steps, burn_in, save_results, save_full_chain)
        println()
        println("Completed SneasyCH4-H MCMC calibration.")
    end

    if magicc
        println("Beginning MCMC calibration of SneasyCH4-M.")
        include("../calibration/sneasy_magicc/climassim.jl")
        run_mcmc_h(mcmc_file, n_steps, burn_in, save_results, save_full_chain)
        println()
        println("Completed SneasyCH4-M MCMC calibration.")
    end
end



##############################################################################################
# TODO: Add description of what this does with function argument descriptions

function run_climate_all(mcmc_data, chain_size::String, save_file::Bool, fund::Bool, hector::Bool, magicc::Bool)

    if fund
        println("Beginning SneasyCH4-F runs.")
        include("../model_results/climate_ar1/sneasy_fund/run_climate_ar1.jl")
        run_climate_f(mcmc_data, chain_size, save_file)
        println()
        println("Completed all SneasyCH4-F runs.")
    end

    if hector
        println()
        println("Beginning SneasyCH4-H runs.")
        include("../model_results/climate_ar1/sneasy_hector/run_climate_ar1.jl")
        run_climate_h(mcmc_data, chain_size, save_file)
        println()
        println("Completed all SneasyCH4-H runs.")
    end

    if magicc
        println()
        println("Beginning SneasyCH4-M runs.")
        include("../model_results/climate_ar1/sneasy_magicc/run_climate_ar1.jl")
        run_climate_m(mcmc_data, chain_size, save_file)
        println()
        println("Completed all SneasyCH4-M runs.")
    end
end

##############################################################################################
# TODO: Add description of what this does with function argument descriptions

function run_scch4_all(mcmc_data::String, constant::Bool, discount_rate::Float64, elast::Float64, prtp::Float64, iam_fund::Bool, iam_dice::Bool, sneasy_f::Bool, sneasy_h::Bool, sneasy_m::Bool, save_file::Bool)

    include("../model_results/scch4/run_scch4.jl")

    if iam_fund
        if sneasy_f
            println()
            println("Beginning SC-CH4 calculations using FUND with SneasyCH4-F climate output.")
            run_scch4(mcmc_data, "sneasy_fund", :FUND, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end

        if sneasy_h
            println()
            println("Beginning SC-CH4 calculations using FUND with SneasyCH4-H climate output.")
            run_scch4(mcmc_data, "sneasy_hector", :FUND, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end

        if sneasy_m
            println()
            println("Beginning SC-CH4 calculations using FUND with SneasyCH4-M climate output.")
            run_scch4(mcmc_data, "sneasy_magicc", :FUND, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end
    end

    if iam_dice
        if sneasy_f
            println()
            println("Beginning SC-CH4 calculations using DICE2013 with SneasyCH4-F climate output.")
            run_scch4(mcmc_data, "sneasy_fund", :DICE, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end

        if sneasy_h
            println()
            println("Beginning SC-CH4 calculations using DICE2013 with SneasyCH4-H climate output.")
            run_scch4(mcmc_data, "sneasy_hector", :DICE, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end

        if sneasy_m
            println()
            println("Beginning SC-CH4 calculations using DICE2013 with SneasyCH4-M climate output.")
            run_scch4(mcmc_data, "sneasy_magicc", :DICE, constant, discount_rate, elast, prtp, save_file)
            println()
            println("Finished SC-CH4 calculations for this specification.")
        end
    end
end
