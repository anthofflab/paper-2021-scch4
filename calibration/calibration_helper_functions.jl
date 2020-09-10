# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file contains functions that are used for the model calibrations.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------



#######################################################################################################################
# LOAD AND CLEAN UP DATA USED FOR MODEL CALIBRATION.
#######################################################################################################################
# Description: This function loads, cleans up, and merges all of the calibration data into a single dataframe.
#
# Function Arguments:
#
#       model_end = The final year to include in the calibration data set (defaults to start in 1765).
#----------------------------------------------------------------------------------------------------------------------

function load_calibration_data(model_end::Int)

    # Set model start year to 1765 by default.
    model_start = 1765

    # Create column of calibration years and calculate indicies for calibration time period (relative to 1765-2018).
    df = DataFrame(year = collect(1765:2018))
    model_calibration_indices = findall((in)(collect(model_start:model_end)), collect(1765:2018))


    #-------------------------------------------------------------------
    # HadCRUT4 temperature data (anomalies relative to 1861-1880 mean).
    #-------------------------------------------------------------------

    # Load raw temperature data.
    raw_temp_data = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "global_temp_hadcrut4.csv"), skiplines_begin=24))

    # Find indices to normalize temperature data to 1861-1880 mean.
    hadcrut_norm_indices = findall((in)(collect(1861:1880)), collect(1850:2017))

    # Normalize temperature data to 1861-1880 mean.
    norm_temp_data  = DataFrame(year=raw_temp_data[!,:year], hadcrut_temperature_obs = raw_temp_data[!,:median] .- mean(raw_temp_data[hadcrut_norm_indices, :median]))

    # Join data on year.
    df = join(df, norm_temp_data, on=:year, kind=:outer)

    # Read in HadCRUT4 1σ errors and rename column.
    raw_temp_errors  = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "global_temp_hadcrut4_1sigma_uncertainty.csv"), skiplines_begin=21))
    rename!(raw_temp_errors, :one_sigma_all => :hadcrut_temperature_sigma)

    # Join data on year.
    df = join(df, raw_temp_errors[!, [:year, :hadcrut_temperature_sigma]], on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Annual Global Ocean Heat Content (0-3000 m).
    #---------------------------------------------------------------------------------

    # Load ocean heat content (0-3000m) observations and errors.
    ocean_heat_raw = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ocean_heat_gouretski_3000m.csv"), colnames=["year", "ocean_heat_obs", "ocean_heat_sigma"], skiplines_begin=3))

    # Join data on year.
    df = join(df, ocean_heat_raw, on=:year, kind=:outer)


    #--------------------------------------------------------
    # Mauna Loa Instrumental Atmospheric CO₂ Concentrations.
    #--------------------------------------------------------

    # Load Mauna Loa CO₂ observations and errors, and rename columns.
    raw_mauna_loa_co2_data  = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "co2_mauna_loa.csv"), skiplines_begin=59))
    rename!(raw_mauna_loa_co2_data, :mean => :maunaloa_co2_obs, :unc => :maunaloa_co2_sigma)

    # Join data on year.
    df = join(df, raw_mauna_loa_co2_data, on=:year, kind=:outer)


    #-----------------------------------------------------
    # Law Dome Ice Core Atmospheric CO₂ Concentrations.
    #-----------------------------------------------------

    # Load Law Dome CO₂ observations and errors, and rename columns.
    raw_law_dome_co2_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "law_dome_co2.csv"), skiplines_begin=4))
    rename!(raw_law_dome_co2_data, :co2_ice => :lawdome_co2_obs, :one_sigma_error => :lawdome_co2_sigma)

    # Join data on year.
    df = join(df, raw_law_dome_co2_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Decadal Ocean Carbon Fluxes
    #---------------------------------------------------------------------------------

    # Observations and errors from McNeil et al. (2003).
    ocean_co2_flux_data = DataFrame(year=[1985, 1995], oceanco2_flux_obs=[1.6, 2.0], oceanco2_flux_sigma=[0.4, 0.4])

    # Join data on year.
    df = join(df, ocean_co2_flux_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # NOAA Instrumental Atmospheric CH₄ Concentrations.
    #---------------------------------------------------------------------------------

    # Load NOAA CH₄ observations and errors, and rename columns.
    raw_noaa_ch4_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ch4_noaa.csv"), skiplines_begin=58))
    rename!(raw_noaa_ch4_data, :mean => :noaa_ch4_obs, :unc => :noaa_ch4_sigma)

    # Join data on year.
    df = join(df, raw_noaa_ch4_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Law Dome Ice Core Atmospheric CH₄ Concentrations.
    #---------------------------------------------------------------------------------

    # Load Law Dome CH₄ observations and dates.
    raw_law_dome_ch4_data = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "ch4_ice_core_etheridge_1998.csv"), skiplines_begin=14))

    # Get all listed years and number of unique years (multiple observations may occur for a single year).
    years = trunc.(Int, raw_law_dome_ch4_data.air_age)
    unique_years = unique(raw_law_dome_ch4_data.air_age)

    # Create array to hold annual CH₄ values.
    n_years = length(unique_years)
    law_dome_ch4_annual_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    # For years with multiple observations, calculate mean CH₄ value.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findall(unique_years[t] .== years)
        #Assign year and mean value for array with one value per year.
        law_dome_ch4_annual_avg[t,:year] = unique_years[t]
        law_dome_ch4_annual_avg[t,:ch4]  = mean(raw_law_dome_ch4_data[index, :CH4])
    end

    # Load CH₄ ice firn observations and dates (also Etheridge 1998).
    raw_ch4_firn_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ch4_firn_etheridge_1998.csv"), skiplines_begin=14))

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc.(Int, raw_ch4_firn_data[:,:air_age])
    unique_years = unique(years)

    #Create array to hold annual CH₄ ice firn values.
    n_years = length(unique_years)
    firn_ch4_annual_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findall(unique_years[t] .== years)
        #Assign year and mean value for array with one value per year.
        firn_ch4_annual_avg[t,:year] = unique_years[t]
        firn_ch4_annual_avg[t,:ch4] = mean(raw_ch4_firn_data[index, :CH4])
    end

    # Merge data for ice core (1852-1977) and firn (1978-1981) that occur during calibration period. After that, calibration uses NOAA flask data.
    core_indices = findall((in)(collect(1852:1977)), law_dome_ch4_annual_avg[:,:year])
    firn_indices = findall((in)(collect(1978:1981)), firn_ch4_annual_avg[:,:year])
    ch4_ice = vcat(law_dome_ch4_annual_avg[core_indices,:], firn_ch4_annual_avg[firn_indices,:])

    # Do a linear interpolation between interpolar (N-S) differences as described in Etheridge et al. (1998).
    # Uses reported Greenland-Antarctica difference of ≈ 41.9 ppb for 1842 and 143 ppb for 1980s (assumed centered on 1985).
    start_year = 1842
    end_year   = 1985
    n_years    = length(start_year:end_year)

    # Initialize array to hold merged, re-scaled global CH₄ data.
    interp_vals = zeros(n_years, 2)
    interp_vals[:,1] = collect(start_year:end_year)

    # Set initial value.
    interp_vals[1,2] = 41.9

    #Calculate annual amount to add for each year in interpolation.
    adder = (143-41.9) / (length(start_year:end_year)-1)

    # Calculate interpolar values to be added to ice core data over time if we had annually consecutive observations.
    for t in 2:n_years
        interp_vals[t,2] = interp_vals[t-1,2] + adder
    end

    #Find indices of annual interpolated interpolar differences that correspond to ice core CH₄ observations.
    interp_index = findall((in)(ch4_ice[!,:year]), interp_vals[:,1])

    # Approximate global CH4 concentration as Antartica ice core values plus 37% of interpolar difference following Etheridge et al. (1998).
    ch4_ice[!,:global_ch4] = ch4_ice[!,:ch4] .+ interp_vals[interp_index, 2] .* 0.37

    # Set observation error for CH₄ ice core data as 15 ppb.
    # Etheridge notes 5ppb (1σ) measurement error and global calculation (using interhemispheric mixing assumptions) could introduce an error of ≈ 10ppb.
    ch4ice_error = ones(size(ch4_ice)[1]) .* 15.0
    final_law_dome_ch4_data = DataFrame(year=ch4_ice[!,:year],  lawdome_ch4_obs = ch4_ice[!,:global_ch4], lawdome_ch4_sigma = ch4ice_error)

    # Join re-scaled global CH₄ observations with rest of calibration data.
    df = join(df, final_law_dome_ch4_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Finalize Joint Calibration Data Set.
    #---------------------------------------------------------------------------------

    # Sort all calibration data by year.
    sort!(df, :year)

    # Crop data to appropriate calibration years and return.
    return df[model_calibration_indices, :]
end



#######################################################################################################################
# BRIDGE SAMPLING FOR MODEL MARGINAL LIKELIHOODS.
#######################################################################################################################
# Description: This function carries out a single bridge sampling iteration for calculating the marginal likelihoods
#              of the four versions of SNEASY+CH4. It closely follows the code from "Neglecting model structural
#              uncertainty underestimates upper tails of flood hazard (Wong et al., 2018).
#
# Function Arguments:
#
#       norm_constant               = Log normalizing constant estimate (passed in as the bridge sampling estimate from previous iteration).
#       mcmc_posterior              = Vector of log-posterior estimates evaluated for each calibrated posterior parameter sample (from MCMC calibration).
#       mcmc_importance_density     = Density of posterior parameter samples (from MCMC calibration) evaluated on the importance distribution.
#       importance_sample_posterior = Vector of log-posterior estimates evaluated for each sample from the importance distribution.
#       importance_sample_density   = Density of importance distribution samples evaluated on the importance distribution.
#----------------------------------------------------------------------------------------------------------------------

function bridge_sampling_iteration(norm_constant, mcmc_posterior, mcmc_importance_density, importance_sample_posterior, importance_sample_density)

    # Filter out non-finite values and normalize finite values based on normalizing constant (i.e. previous bridge sampling estimate).
    mcmc_posterior_norm = mcmc_posterior[isfinite.(mcmc_posterior)] .- norm_constant
    importance_sample_posterior_norm = importance_sample_posterior[isfinite.(importance_sample_posterior)] .- norm_constant

    mcmc_importance_density = mcmc_importance_density[isfinite.(mcmc_posterior)]
    importance_sample_density = importance_sample_density[isfinite.(importance_sample_posterior)]

    # Get number of samples with finite values.
    n_mcmc = length(mcmc_posterior_norm)
    n_importance = length(importance_sample_posterior_norm)

    # Compute updated estimates for numerator and denominator means.
    # Note: A small subset of cases produce non-finite values that lead to an NaN result. First calculate temporary variables with these potential NaNs and then filter them out.
    importance_temp = exp.(importance_sample_posterior_norm) ./ (n_importance .* exp.(importance_sample_density) .+ n_mcmc .* exp.(importance_sample_posterior_norm))
    mcmc_temp       = exp.(mcmc_importance_density) ./ (n_importance .* exp.(mcmc_importance_density) .+ n_mcmc .* exp.(mcmc_posterior_norm))

    # Calcualte updated mean values for non-NaN results.
    importance_mean =  mean(importance_temp[findall(x -> !isnan(x), importance_temp)])
    mcmc_mean       =  mean(mcmc_temp[findall(x -> !isnan(x), mcmc_temp)])

    # Return updated bridge sampling estimate.
    return (norm_constant + log(importance_mean) - log(mcmc_mean))
end



#######################################################################################################################
# CALCULATE BAYESIAN MODEL AVERAGING (BMA) WEIGHTS USING BRIDGE SAMPLING.
#######################################################################################################################
# Description: This function uses bridge sampling to calculate the marginal likelihoods of the four versions of
#              SNEASY+CH4. It uses a multivariate normal approximation to the joint posterior distribtion as the
#              importance distribution and closely follows the code from "Neglecting model structural uncertainty
#              underestimates upper tails of flood hazard (Wong et al., 2018).
#
# Function Arguments:
#
#       parameters_fairch4      = Calibrated parameters for SNEASY-FAIR (each row is a new sample from joint posterior distribution, each column is a different parameter).
#       parameters_fundch4      = Calibrated parameters for SNEASY-FUND (each row is a new sample from joint posterior distribution, each column is a different parameter).
#       parameters_hectorch4    = Calibrated parameters for SNEASY-Hector (each row is a new sample from joint posterior distribution, each column is a different parameter).
#       parameters_magiccch4    = Calibrated parameters for SNEASY-MAGICC (each row is a new sample from joint posterior distribution, each column is a different parameter).
#       log_posterior_fairch4   = Function to calculate log-posterior for SNEASY-FAIR.
#       log_posterior_fundch4   = Function to calculate log-posterior for SNEASY-FUND.
#       log_posterior_hectorch4 = Function to calculate log-posterior for SNEASY-Hector.
#       log_posterior_magiccch4 = Function to calculate log-posterior for SNEASY-MAGICC.
#----------------------------------------------------------------------------------------------------------------------

function calculate_bma_weights(parameters_fairch4::Array{Float64,2}, parameters_fundch4::Array{Float64,2}, parameters_hectorch4::Array{Float64,2}, parameters_magiccch4::Array{Float64,2}, log_posterior_fairch4, log_posterior_fundch4, log_posterior_hectorch4, log_posterior_magiccch4)

    # Calculate number of posterior parameter samples.
    n_samples = size(parameters_fairch4, 1)

    # Pre-allocate arrays to store log-posterior values resulting from sampled posterior and importance pdf model parameters.
    posterior_mcmc_fair   = zeros(n_samples)
    posterior_mcmc_fund   = zeros(n_samples)
    posterior_mcmc_hector = zeros(n_samples)
    posterior_mcmc_magicc = zeros(n_samples)

    posterior_importance_fair   = zeros(n_samples)
    posterior_importance_fund   = zeros(n_samples)
    posterior_importance_hector = zeros(n_samples)
    posterior_importance_magicc = zeros(n_samples)

    # Calculate log-posterior values resulting from calibrated posterior model parameters.
    for i = 1:n_samples
        posterior_mcmc_fair[i]   = log_posterior_fairch4(parameters_fairch4[i,:])
        posterior_mcmc_fund[i]   = log_posterior_fundch4(parameters_fundch4[i,:])
        posterior_mcmc_hector[i] = log_posterior_hectorch4(parameters_hectorch4[i,:])
        posterior_mcmc_magicc[i] = log_posterior_magiccch4(parameters_magiccch4[i,:])
    end

    # Calculate posterior parameter mean and covariance matrix, then fit to a multivariate normal distribution (importance pdf).
    posterior_mean_fair = vec(mean(parameters_fairch4, dims=1))
    posterior_cov_fair  = cov(parameters_fairch4)
    fair_importance_pdf = MvNormal(posterior_mean_fair, posterior_cov_fair)

    posterior_mean_fund = vec(mean(parameters_fundch4, dims=1))
    posterior_cov_fund  = cov(parameters_fundch4)
    fund_importance_pdf = MvNormal(posterior_mean_fund, posterior_cov_fund)

    posterior_mean_hector = vec(mean(parameters_hectorch4, dims=1))
    posterior_cov_hector  = cov(parameters_hectorch4)
    hector_importance_pdf = MvNormal(posterior_mean_hector, posterior_cov_hector)

    posterior_mean_magicc = vec(mean(parameters_magiccch4, dims=1))
    posterior_cov_magicc  = cov(parameters_magiccch4)
    magicc_importance_pdf = MvNormal(posterior_mean_magicc, posterior_cov_magicc)

    # Calculate importance densities using calibrated posterior parameters.
    importance_mcmc_density_fair   = logpdf(fair_importance_pdf,   transpose(parameters_fairch4))
    importance_mcmc_density_fund   = logpdf(fund_importance_pdf,   transpose(parameters_fundch4))
    importance_mcmc_density_hector = logpdf(hector_importance_pdf, transpose(parameters_hectorch4))
    importance_mcmc_density_magicc = logpdf(magicc_importance_pdf, transpose(parameters_magiccch4))

    # Sample parameters from fitted importance pdf.
    importance_sample_fair   = rand(fair_importance_pdf,   n_samples)
    importance_sample_fund   = rand(fund_importance_pdf,   n_samples)
    importance_sample_hector = rand(hector_importance_pdf, n_samples)
    importance_sample_magicc = rand(magicc_importance_pdf, n_samples)

    # Calculate importance densities using samples from importance pdf.
    importance_sample_density_fair   = logpdf(fair_importance_pdf,   importance_sample_fair)
    importance_sample_density_fund   = logpdf(fund_importance_pdf,   importance_sample_fund)
    importance_sample_density_hector = logpdf(hector_importance_pdf, importance_sample_hector)
    importance_sample_density_magicc = logpdf(magicc_importance_pdf, importance_sample_magicc)

    # Calculate log-posterior values of samples from importance pdf.
    # Note: A small subset of samples from importance pdf produce non-physical model results, so also set them as -Inf to be filtered out.

    # SNEASY-FAIR
    for i = 1:length(posterior_importance_fair)
        try
            posterior_importance_fair[i] = log_posterior_fairch4(importance_sample_fair[:,i])
        catch
            posterior_importance_fair[i] = -Inf
        end
    end

    # SNEASY-FUND
    for i = 1:length(posterior_importance_fund)
        try
            posterior_importance_fund[i] = log_posterior_fundch4(importance_sample_fund[:,i])
        catch
            posterior_importance_fund[i] = -Inf
        end
    end

    # SNEASY-Hector
    for i = 1:length(posterior_importance_hector)
        try
            posterior_importance_hector[i] = log_posterior_hectorch4(importance_sample_hector[:,i])
        catch
            posterior_importance_hector[i] = -Inf
        end
    end

    # SNEASY-MAGICC
    for i = 1:length(posterior_importance_magicc)
        try
            posterior_importance_magicc[i] = log_posterior_magiccch4(importance_sample_magicc[:,i])
        catch
            posterior_importance_magicc[i] = -Inf
        end
    end


    # Set convergence tolerance and maximum number of bridge-sampling iterations for all models.
    max_iterations = 10_000
    tolerance = 1e-10

    # Set initial conditions to begin iterating over for all models (following Wong et al., 2018, set initial value by "averaging the ratios on a log scale.").
    x_fair   = [0.0, -mean(importance_mcmc_density_fair   - posterior_mcmc_fair)]
    x_fund   = [0.0, -mean(importance_mcmc_density_fund   - posterior_mcmc_fund)]
    x_hector = [0.0, -mean(importance_mcmc_density_hector - posterior_mcmc_hector)]
    x_magicc = [0.0, -mean(importance_mcmc_density_magicc - posterior_mcmc_magicc)]

    #-------------------------------------------------------------
    # Use bridge sampling to estimate model marginal likelihoods.
    #-------------------------------------------------------------

    # Iterate over SNEASY-FAIR values until tolerance or maximum iterations reached.
    let iterations = 0
        while abs(x_fair[2] - x_fair[1]) >= tolerance && iterations <= max_iterations
            # Set x[1] to previous iteration's estimated value.
            x_fair[1] = x_fair[2]
            # Calculate new x[2] value for current iteration.
            x_fair[2] = bridge_sampling_iteration(x_fair[1], posterior_mcmc_fair, importance_mcmc_density_fair, posterior_importance_fair, importance_sample_density_fair)
            # Add one to iteration.
            iterations += 1
        end
    end

    # Iterate over SNEASY-FUND values until tolerance or maximum iterations reached.
    let iterations = 0
        while abs(x_fund[2] - x_fund[1]) >= tolerance && iterations <= max_iterations
            # Set x[1] to previous iteration's estimated value.
            x_fund[1] = x_fund[2]
            # Calculate new x[2] value for current iteration.
            x_fund[2] = bridge_sampling_iteration(x_fund[1], posterior_mcmc_fund, importance_mcmc_density_fund, posterior_importance_fund, importance_sample_density_fund)
            # Add one to iteration.
            iterations += 1
        end
    end

    # Iterate over SNEASY-Hector values until tolerance or maximum iterations reached.
    let iterations = 0
        while abs(x_hector[2] - x_hector[1]) >= tolerance && iterations <= max_iterations
            # Set x[1] to previous iteration's estimated value.
            x_hector[1] = x_hector[2]
            # Calculate new x[2] value for current iteration.
            x_hector[2] = bridge_sampling_iteration(x_hector[1], posterior_mcmc_hector, importance_mcmc_density_hector, posterior_importance_hector, importance_sample_density_hector)
            # Add one to iteration.
            iterations += 1
        end
    end

    # Iterate over SNEASY-MAGICC values until tolerance or maximum iterations reached.
    let iterations = 0
        while abs(x_magicc[2] - x_magicc[1]) >= tolerance && iterations <= max_iterations
            # Set x[1] to previous iteration's estimated value.
            x_magicc[1] = x_magicc[2]
            # Calculate new x[2] value for current iteration.
            x_magicc[2] = bridge_sampling_iteration(x_magicc[1], posterior_mcmc_magicc, importance_mcmc_density_magicc, posterior_importance_magicc, importance_sample_density_magicc)
            # Add one to iteration.
            iterations += 1
        end
    end

    # Collect marginal likelihoods of each model estimated from bridge sampling approach.
    model_marginal_likelihood = [x_fair[2], x_fund[2], x_hector[2], x_magicc[2]]

    # Normalize model likelihoods relative to largest value.
    normalized_model_marginal_likelihood = model_marginal_likelihood .- maximum(model_marginal_likelihood)

    # Use model marginal likelihoods from bridge sampling to calculate BMA weights.
    bma_weights = exp.(normalized_model_marginal_likelihood) ./ sum(exp.(normalized_model_marginal_likelihood))

    return bma_weights
end

#######################################################################################################################
# CALCULATE AR(1) LOG-LIKELIHOOD.
########################################################################################################################
# Description: This function calculates the AR(1) log-likelihood in terms of the data-model residuls when accounting for
#              time-varying observation errors. It follows "The Effects of Time-Varying Observation Errors on Semi-Empirical
#              Sea-Level Projections" (Ruckert et al., 2017) DOI 10.1007/s10584-016-1858-z.
#
# Function Arguments:
#
#       residuals = A vector of data-model residuals.
#       σ         = AR(1) innovation standard deviation.
#       ρ         = AR(1) autocorrelation term.
#       ϵ         = A vector of time-varying observation error estimates (from calibration data sets).
#----------------------------------------------------------------------------------------------------------------------

function hetero_logl_ar1(residuals::Array{Float64,1}, σ::Float64, ρ::Float64, ϵ::Array{Union{Float64, Missings.Missing},1})

    # Calculate length of residuals.
    n=length(residuals)

    # Define AR(1) stationary process variance.
    σ_process = σ^2/(1-ρ^2)

    # Initialize AR(1) covariance matrix (just for convenience).
    H = abs.(collect(1:n)' .- collect(1:n))

    # Calculate residual covariance matrix (sum of AR(1) process variance and observation error variances).
    # Note: This follows Supplementary Information Equation (10) in Ruckert et al. (2017).
    cov_matrix = σ_process * ρ .^ H + Diagonal(ϵ.^2)

    # Return the log-likelihood.
    return logpdf(MvNormal(cov_matrix), residuals)
end



#######################################################################################################################
# CALCULATE CAR(1) LOG-LIKELIHOOD.
########################################################################################################################
# Description: This function calculates the continuous time autoregressive, or CAR(1), log-likelihood for irregularly
#              spaced data in terms of the data-model residuls when accounting for time-varying observation errors. It
#              builds off of "The Effects of Time-Varying Observation Errors on Semi-Empirical Sea-Level Projections"
#              (Ruckert et al., 2017) DOI 10.1007/s10584-016-1858-z and "The Analysis of Irregularly Observed Stochastic
#              Astronomical Time-Series – I. Basics of Linear Stochastic Differential Equations" (Koen, 2005)
#              doi.org/10.1111/j.1365-2966.2005.09213.x
#
# Function Arguments:
#
#       residuals      = A vector of data-model residuals.
#       indices        = Index positions of observations relative to model time horizon (i.e. the first model time period = 1, the second = 2, etc.).
#       σ²_white_noise = Variance of the continuous white noise process.
#       α₀             = Parameter describing correlation memory of CAR(1) process.
#       ϵ              = A vector of time-varying observation error estimates (from calibration data sets).
#----------------------------------------------------------------------------------------------------------------------

function hetero_logl_car1(residuals::Array{Float64,1}, indices::Array{Int,1}, σ²_white_noise::Float64, α₀::Float64, ϵ::Array{Union{Float64, Missings.Missing},1})

    # Calculate length of residuals.
    n=length(residuals)

    # Initialize covariance matrix for irregularly spaced data with relationships decaying exponentially.
    H = exp.(-α₀ .* abs.(indices' .- indices))

    # Define the variance of x(t), a continous stochastic time-series.
    σ² = σ²_white_noise / (2*α₀)

    # Calculate residual covariance matrix (sum of CAR(1) process variance and observation error variances).
    cov_matrix = σ² .* H + Diagonal(ϵ.^2)

    # Return the log-likelihood.
    return logpdf(MvNormal(cov_matrix), residuals)
end
