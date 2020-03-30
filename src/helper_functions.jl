# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file contains functions that are used for various climate projection and SC-CH4 calculations.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------



#######################################################################################################################
# CREATE FOLDER STRUCTURE TO STORE RESULTS.
#######################################################################################################################
# Description: This function creates a directory structure to store all of the model output in the "Results" folder.
#              The main folder will have the user assigned "results_name".
#
# Function Arguments:
#
#       results_name = A vector of global carbon tax values to be optimized.
#----------------------------------------------------------------------------------------------------------------------

function build_result_folders(results_name::String)

    # Set SNEASY+CH4 and IAM model names.
    climate_model_names = ["s_fair", "s_fund", "s_hector", "s_magicc"]
    iam_model_names     = ["dice", "fund"]

    # Loop over the various IAM and climate model combinations.
    for climate_model in climate_model_names

        # Create folders to store calibrated posterior parameter samples for baseline and sensitivity runs.
        mkpath(joinpath("results", results_name, "calibrated_parameters", climate_model))
        mkpath(joinpath("results", results_name, "sensitivity_results", "wider_priors", "calibrated_parameters", climate_model))
        mkpath(joinpath("results", results_name, "sensitivity_results", "alternative_ch4_emissions", "calibrated_parameters", climate_model))

        # Create folders to store climate projections for the different scenarios (and baseline sensitivity runs).
        mkpath(joinpath("results", results_name, "climate_projections", "baseline_run", climate_model))
        mkpath(joinpath("results", results_name, "sensitivity_results", "wider_priors", "climate_projections", "baseline_run", climate_model))
        mkpath(joinpath("results", results_name, "sensitivity_results", "alternative_ch4_emissions", "climate_projections", "baseline_run", climate_model))

        mkpath(joinpath("results", results_name, "climate_projections", "outdated_forcing", climate_model))
        mkpath(joinpath("results", results_name, "climate_projections", "remove_correlations", climate_model))
        mkpath(joinpath("results", results_name, "climate_projections", "us_climate_sensitivity", climate_model))
        mkpath(joinpath("results", results_name, "climate_projections", "rcp26", climate_model))

        # Create folders to store SC-CH4 estimates for DICE and FUND for baseline, alternative scenario, and sensitivity climate runs.
        for iam_model in iam_model_names
            mkpath(joinpath("results", results_name, "scch4_estimates", "baseline_run", iam_model, climate_model))
            mkpath(joinpath("results", results_name, "sensitivity_results", "wider_priors", "scch4_estimates", "baseline_run", iam_model, climate_model))
            mkpath(joinpath("results", results_name, "sensitivity_results", "alternative_ch4_emissions", "scch4_estimates", "baseline_run", iam_model, climate_model))

            mkpath(joinpath("results", results_name, "scch4_estimates", "outdated_forcing", iam_model, climate_model))
            mkpath(joinpath("results", results_name, "scch4_estimates", "remove_correlations", iam_model, climate_model))
            mkpath(joinpath("results", results_name, "scch4_estimates", "us_climate_sensitivity", iam_model, climate_model))
            mkpath(joinpath("results", results_name, "scch4_estimates", "rcp26", iam_model, climate_model))
        end

        # Create folders to save equity-weighted SC-CH4 estimatesfor FUND and RICE.
        mkpath(joinpath("results", results_name, "scch4_estimates", "equity_weighting", "fund", climate_model))
        mkpath(joinpath("results", results_name, "scch4_estimates", "equity_weighting", "rice", climate_model))
    end

    # Create folder to store Bayesian model averaging (BMA) weights within the calibration folder.
    mkpath(joinpath("results", results_name, "calibrated_parameters", "bma_weights"))
    mkpath(joinpath("results", results_name, "sensitivity_results", "wider_priors", "calibrated_parameters", "bma_weights"))
    mkpath(joinpath("results", results_name, "sensitivity_results", "alternative_ch4_emissions", "calibrated_parameters", "bma_weights"))

end



#######################################################################################################################
# CREATE A FUNCTION TO UPDATE PARAMETERS FOR SPECIFIC METHANE CYCLE MODEL.
#######################################################################################################################
# Description: This function will update the uncertain parameters specific to the methane cycle being used.
#
# Function Arguments:
#
#       climate_model = A symbol identifying the specific version of SNEASY+CH4 (options are :sneasy_fair, :sneasy_fund,
#                       :sneasy_hector, and :sneasy_magicc).
#----------------------------------------------------------------------------------------------------------------------

function create_update_ch4_function(climate_model::Symbol)

    # Create a function to update the uncertain CH₄ parameters, given the selected CH₄ model.
    update_ch4_params! =

        if climate_model == :sneasy_fair

            function(m::Model, params::Array{Float64,1})
                CH₄_0         = params[12]
                N₂O_0         = params[13]
                rf_scale_CH₄  = params[17]
                τ_troposphere = params[22]
                CH₄_natural   = params[23]

                update_param!(m, :natural_emiss_CH₄, ones(length(dim_keys(m, :time))) .* CH₄_natural)
                update_param!(m, :CH₄_0,             CH₄_0)
                update_param!(m, :τ_CH₄,             τ_troposphere)
                update_param!(m, :N₂O_0,             N₂O_0)
                update_param!(m, :scale_CH₄,         rf_scale_CH₄)
                return
            end

        elseif climate_model == :sneasy_fund

            function(m::Model, params::Array{Float64,1})
                CH₄_0         = params[12]
                N₂O_0         = params[13]
                rf_scale_CH₄  = params[17]
                τ_troposphere = params[22]

                update_param!(m, :ch4pre,    CH₄_0)
                update_param!(m, :acch4_0,   CH₄_0)
                update_param!(m, :lifech4,   τ_troposphere)
                update_param!(m, :CH₄_0,     CH₄_0)
                update_param!(m, :N₂O_0,     N₂O_0)
                update_param!(m, :scale_CH₄, rf_scale_CH₄)
                return
            end

        elseif climate_model == :sneasy_hector

            function(m::Model, params::Array{Float64,1})
                CH₄_0          = params[12]
                N₂O_0          = params[13]
                rf_scale_CH₄   = params[17]
                τ_troposphere  = params[22]
                CH₄_natural    = params[23]
                τ_soil         = params[24]
                τ_stratosphere = params[25]

                update_param!(m, :TOH0,      τ_troposphere)
                update_param!(m, :M0,        CH₄_0)
                update_param!(m, :CH4N,      CH₄_natural)
                update_param!(m, :Tsoil,     τ_soil)
                update_param!(m, :Tstrat,    τ_stratosphere)
                update_param!(m, :CH₄_0,     CH₄_0)
                update_param!(m, :N₂O_0,     N₂O_0)
                update_param!(m, :scale_CH₄, rf_scale_CH₄)
                return
            end

        elseif climate_model == :sneasy_magicc

            function(m::Model, params::Array{Float64,1})
                CH₄_0          = params[12]
                N₂O_0          = params[13]
                rf_scale_CH₄   = params[17]
                τ_troposphere  = params[22]
                CH₄_natural    = params[23]
                τ_soil         = params[24]
                τ_stratosphere = params[25]

                update_param!(m, :CH₄_0,       CH₄_0)
                update_param!(m, :CH4_natural, CH₄_natural)
                update_param!(m, :TAUSOIL,     τ_soil)
                update_param!(m, :TAUSTRAT,    τ_stratosphere)
                update_param!(m, :TAUINIT,     τ_troposphere)
                update_param!(m, :N₂O_0,       N₂O_0)
                update_param!(m, :scale_CH₄,   rf_scale_CH₄)
                return
            end

        else
            error("Incorrect climate model selected. Options include :sneasy_fair, :sneasy_fund, :sneasy_hector, or :sneasy_magicc.")
        end

    # Return the paramter updated function for the specific version of SNEASY+CH4.
    return update_ch4_params!
end



#######################################################################################################################
# CREATE A FUNCTION TO ACCESS PROJECTED METHANE CONCENTRATIONS FOR SPECIFIC METHANE CYCLE MODEL.
#######################################################################################################################
# Description: This function will access the CH₄ concentration projections given the particular version of
#              SNEASY+CH4 (component and variable names differ across model versions).
#
# Function Arguments:
#
#       climate_model = A symbol identifying the specific version of SNEASY+CH4 (options are :sneasy_fair, :sneasy_fund,
#                       :sneasy_hector, and :sneasy_magicc).
#----------------------------------------------------------------------------------------------------------------------

function create_get_ch4_results_function(climate_model::Symbol)

    # Create a function to get CH₄ concentration projections, given the selected CH₄ model.
    get_ch4_results! =

        if climate_model == :sneasy_fair || climate_model == :sneasy_magicc

            function(m::Model)
                return m[:ch4_cycle, :CH₄]
            end

        elseif climate_model == :sneasy_fund

            function(m::Model)
                return m[:climatech4cycle, :acch4]
            end

        elseif climate_model == :sneasy_hector

            function(m::Model)
                return m[:ch4_cycle, :CH4]
            end
        end

    return get_ch4_results!
end



#######################################################################################################################
# CALCULATE DISCOUNT FACTORS
#######################################################################################################################
# Description: This function calculates discount factors given user-specified discounting settings.
#
# Function Arguments:
#
#       df               = A pre-allocated array to store discount factors.
#       pc_consumption   = Per capita consumption values.
#       ρ                = Pure rate of time preference.
#       η                = Elasticity of marginal utility of consumption.
#       years            = Vector of years that the model is being run for.
#       pulse_year       = The year that the CH₄ emission pulse is added.
#       pulse_year_index = The timestep index corresponding to the CH₄ emission pulse year.
#----------------------------------------------------------------------------------------------------------------------

function calculate_discount_factors!(df::Array{Float64,1}, pc_consumption::Array{Float64,1}, ρ::Float64, η::Float64, years::Array{Int,1}, pulse_year::Int, pulse_year_index::Int)
    for t=pulse_year_index:length(years)
        df[t] = (pc_consumption[pulse_year_index] / pc_consumption[t])^η * 1.0 / (1.0 + ρ)^(years[t]-pulse_year)
    end
end



#######################################################################################################################
# CALCULATE CO₂ RADIATIVE FORCING SCALAR
#######################################################################################################################
# Description: This function calculates a scaling coefficient so CO₂ radiative forcing is consistent with user supplied
#              value for the radiative forcing due to doubling CO₂ (following Smith et al. (2018), Geosci. Model Dev.).
#
# Function Arguments:
#
#       F2x   = Radiative forcing due to doubling carbon dioxide (Wm⁻²).
#       CO₂_0 = Initial (pre-industrial) carbon dioxide concentration (ppm).
#       N₂O_0 = Initial (pre-industrial) nitrous oxide concentration (ppb).
#----------------------------------------------------------------------------------------------------------------------

function co2_rf_scale(F2x::Float64, CO₂_0::Float64, N₂O_0::Float64)

    # Calcualte forcing from doubling of CO₂.
    co2_doubling = (-2.4e-7 * CO₂_0^2 + 7.2e-4 * CO₂_0 - 2.1e-4 * N₂O_0 + 5.36) * log(2)

    # Calculate scaling factor, given user supplied F2x parameter.
    CO₂_scale = F2x / co2_doubling

    return CO₂_scale
end



#######################################################################################################################
# SIMULATE STATIONARY AR(1) PROCESS WITH TIME VARYING OBSERVATION ERRORS.
#######################################################################################################################
# Description: This function simulates a stationary AR(1) process (given time-varying observation errors supplied with
#              each calibration data set) to superimpose noise onto the climate model projections.
#
# Function Arguments:
#
#       n = Number of time periods (years) the model is being run for.
#       σ = Calibrated standard deviation.
#       ρ = Calibrated autocorrelation coefficient.
#       ϵ = Time-varying observation errors.
#----------------------------------------------------------------------------------------------------------------------

function simulate_ar1_noise(n::Int, σ::Float64, ρ::Float64, ϵ::Array{Float64,1})

    # Define AR(1) stationary process variance.
    σ_process = σ^2/(1-ρ^2)

    # Initialize AR(1) covariance matrix (just for convenience).
    H = abs.(collect(1:n)' .- collect(1:n))

    # Calculate residual covariance matrix (sum of AR(1) process variance and observation error variances).
    # Note: This follows Supplementary Information Equation (10) in Ruckert et al. (2017).
    cov_matrix = σ_process * ρ .^ H + Diagonal(ϵ.^2)

    # Return a mean-zero AR(1) noise sample accounting for time-varying observation error.
    return rand(MvNormal(cov_matrix))
end



#######################################################################################################################
# SIMULATE STATIONARY CAR(1) PROCESS WITH TIME VARYING OBSERVATION ERRORS.
#######################################################################################################################
# Description: This function simulates a stationary CAR(1) process (given time-varying observation errors supplied with
#              each calibration data set) to superimpose noise onto the climate model projections.
#
# Function Arguments:
#
#       n              = Number of time periods (years) the model is being run for.
#       α₀             = Calibrated term describing correlation memory of CAR(1) process.
#       σ²_white_noise = Calibrated continuous white noise process variance term.
#       ϵ              = Time-varying observation errors.
#----------------------------------------------------------------------------------------------------------------------

function simulate_car1_noise(n, α₀, σ²_white_noise, ϵ)

    # Indices for full time horizon.
    indices = collect(1:n)

    # Initialize covariance matrix for irregularly spaced data with relationships decaying exponentially.
    H = exp.(-α₀ .* abs.(indices' .- indices))

    # Define the variance of x(t), a continous stochastic time-series.
    σ² = σ²_white_noise / (2*α₀)

    # Calculate residual covariance matrix (sum of CAR(1) process variance and observation error variances).
    cov_matrix = σ² .* H + Diagonal(ϵ.^2)

    # Return a mean-zero CAR(1) noise sample accounting for time-varying observation error.
    return rand(MvNormal(cov_matrix))
end



#######################################################################################################################
# REPLICATE TIME-VARYING OBSERVATION ERRORS FOR PERIODS WITHOUT COVERAGE.
#######################################################################################################################
# Description: This function creates a time-series of observation errors for the entire model time horizon. For years
#              without observation error estimates, the error remains constant at the average of the ten nearest error
#              values in time.
#
# Function Arguments:
#
#       start_year = The first year to run the climate model.
#       end_year   = The final year to run the climate model.
#       error_data = A vector of time-varying observation errors supplied with each calibration data set.
#----------------------------------------------------------------------------------------------------------------------

function replicate_errors(start_year::Int, end_year::Int, error_data)

    # Initialize model years and measurement error vector.
    model_years = collect(start_year:end_year)
    errors = zeros(length(model_years))

    # Find indices for periods that have observation errors.
    err_indices = findall(x-> !ismissing(x), error_data)

    # Replicate errors for all periods prior to start of observations based on average of 10 nearest errors in time.
    errors[1:(err_indices[1]-1)] .= mean(error_data[err_indices[1:10]])

    # Add errors for periods with observations.
    errors[err_indices[1:end]] = error_data[err_indices[1:end]]

    # Replicate errors for all periods after observation data based on average of 10 nearest errors in time.
    errors[(err_indices[end]+1):end] .= mean(error_data[err_indices[(end-9):end]])

    return errors
end



#######################################################################################################################
# LINEARLY INTERPOLATE DICE RESULTS TO ANNUAL VALUES
#######################################################################################################################
# Description: This function uses linear interpolation to create an annual time series from DICE results (which have
#              five year timesteps).
#
# Function Arguments:
#
#       data    = The DICE results to be interpolated
#       spacing = Length of model time steps (5 for DICE).
#----------------------------------------------------------------------------------------------------------------------

function dice_interpolate(data, spacing)

    # Create an interpolation object for the data (assume first and last points are end points, e.g. no interpolation beyond support).
    interp_linear = interpolate(data, BSpline(Linear()), OnGrid())

    # Create points to interpolate for (based on spacing term).
    interp_points = collect(1:(1/spacing):length(data))

    # Carry out interpolation.
    return interp_linear[interp_points]
end



#######################################################################################################################
# CREATE CLIMATE PROJECTION CREDIBLE INTERVALS
#######################################################################################################################
# Description: This function calculates the upper and lower credible interval ranges for the climate projections carried
#              out with each posterior parameter sample. It does so for two different % levels.
#
# Function Arguments:
#
#       years          = The years the model projection is carried out for.
#       model_result   = An array of model projections (each row is a new projection, each column is a new year).
#       conf_1_percent = First percentage value to calculate credible interval over (i.e. 0.95 corresponds to 95% credible interval).
#       conf_2_percent = Second percentage value to calculate credible interval over.
#----------------------------------------------------------------------------------------------------------------------

function get_confidence_interval(years, model_result, conf_1_percent, conf_2_percent)

    # Set intervals for quantile function and calculate total number of years in results.
    α1      = 1-conf_1_percent
    α2      = 1-conf_2_percent
    n_years = length(years)

    # Initialize dataframe of results with a column of years, mean results, and missing values.
    ci_results = DataFrame(Year=years, Mean=vec(mean(model_result, dims=1)), Lower1=zeros(Union{Missing,Float64}, n_years), Upper1=zeros(Union{Missing,Float64}, n_years), Lower2=zeros(Union{Missing,Float64}, n_years), Upper2=zeros(Union{Missing,Float64}, n_years))

    # Calculate credible intervals for each year (CI for years with 'missing' values also set to 'missing').
    for i in 1:n_years
        if all(x-> x !== missing, model_result[:,i])
            ci_results.Lower1[i] = quantile(model_result[:,i], α1/2)
            ci_results.Upper1[i] = quantile(model_result[:,i], 1-α1/2)
            ci_results.Lower2[i] = quantile(model_result[:,i], α2/2)
            ci_results.Upper2[i] = quantile(model_result[:,i], 1-α2/2)
        else
            ci_results.Lower1[i] = missing
            ci_results.Upper1[i] = missing
            ci_results.Lower2[i] = missing
            ci_results.Upper2[i] = missing
        end
    end

    # Rename columns to have names specific to user-provided credible interval percentages.
    rename!(ci_results, :Lower1 => Symbol(join(["LowerConf" conf_1_percent], '_')))
    rename!(ci_results, :Upper1 => Symbol(join(["UpperConf" conf_1_percent], '_')))
    rename!(ci_results, :Lower2 => Symbol(join(["LowerConf" conf_2_percent], '_')))
    rename!(ci_results, :Upper2 => Symbol(join(["UpperConf" conf_2_percent], '_')))

    return ci_results
end
