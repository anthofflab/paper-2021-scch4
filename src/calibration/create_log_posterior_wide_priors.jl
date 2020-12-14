# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file contains functions used to calculate the log-posterior using wider prior distributions.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

#######################################################################################################################
# CALCULATE TOTAL (LOG) PRIOR PROBABILITY.
#######################################################################################################################
# Description: This creates a function that will calculates the total (log) prior probability of the uncertain model,
#              initial condition, and statistical process parameters specific to the CH₄ cycle model selected.
#
# Function Arguments:
#
#       climate_model = A symbol identifying the specific version of SNEASY+CH4 (options are :sneasy_fair, :sneasy_fund,
#                       :sneasy_hector, and :sneasy_magicc).
#----------------------------------------------------------------------------------------------------------------------

function construct_log_prior_wider(climate_model::Symbol)

    # Declare all prior distributions for all uncertain parameters found in the four versions of SNEASY+CH4.

    # -----------------------------------------
    # Statistical Process Priors.
    # -----------------------------------------
    prior_σ_temperature      = Uniform(0, 0.24)
    prior_σ_ocean_heat       = Uniform(0, 4.8)
    prior_σ²_white_noise_CO₂ = Uniform(0, 240)
    prior_σ²_white_noise_CH₄ = Uniform(0, 960)

    prior_ρ_temperature      = Uniform(-0.99, 0.99)
    prior_ρ_ocean_heat       = Uniform(-0.99, 0.99)
    prior_α₀_CO₂             = Uniform(0.008, 13.8)
    prior_α₀_CH₄             = Uniform(0.008, 13.8)

    # -----------------------------------------
    # Initial Condition Priors.
    # -----------------------------------------
    prior_temperature_0     = Normal(0, 1.2)
    prior_ocean_heat_0      = Uniform(-120, 0)
    prior_CO₂_0             = Uniform(220, 337.2)
    prior_CH₄_0             = Uniform(552.8, 903.6)
    prior_N₂O_0             = Uniform(211.2, 338.4)

    # -----------------------------------------
    # Climate & Radiative Forcing Priors.
    # -----------------------------------------
    prior_ECS               = Truncated(Cauchy(3.0,2.4), 0.0, 12.0)
    prior_heat_diffusivity  = LogNormal(1.1, 0.36)
    prior_rf_scale_aerosol  = TriangularDist(0., 3.6, 1.)
    prior_rf_scale_CH₄      = Uniform(0.576, 1.536)
    prior_F2x_CO₂           = Uniform(2.376, 5.34)

    # -----------------------------------------
    # Carbon Cycle Priors.
    # -----------------------------------------
    prior_Q10               = Uniform(1.0, 6)
    prior_CO₂_fertilization = Uniform(0., 1.2)
    prior_CO₂_diffusivity   = Uniform(0., 240)

    # -----------------------------------------
    # Methane Cycle Priors.
    # -----------------------------------------
    prior_τ_troposphere     = Uniform(5.36, 16.2)
    prior_CH₄_natural       = Uniform(120, 580.8)
    prior_τ_soil            = Uniform(76.8, 172.8)
    prior_τ_stratosphere    = Uniform(80, 240)


    #----------------------------------------------------------------------------------------------------------------
    # Create a function unique to each CH₄ cycle model to calculate prior distribution for uncertain CH₄ parameters.
    #----------------------------------------------------------------------------------------------------------------

    # CH₄ prior function (requires the particular ordering of uncertain parameters used below).
    ch4_model_priors =

        # Create function based on uncertain SNEASY-FAIR CH₄ cycle parameters.
        if climate_model == :sneasy_fair
            function(p)
                τ_troposphere = p[22]
                CH₄_natural   = p[23]
                return logpdf(prior_τ_troposphere, τ_troposphere) + logpdf(prior_CH₄_natural, CH₄_natural)
            end

        # Create function based on uncertain SNEASY-FUND CH₄ cycle parameters.
        elseif climate_model == :sneasy_fund
            function(p)
                τ_troposphere = p[22]
                return logpdf(prior_τ_troposphere, τ_troposphere)
            end

        # Create function based on uncertain SNEASY-Hector or SNEASY-MAGICC CH₄ cycle parameters.
        elseif climate_model == :sneasy_hector || climate_model == :sneasy_magicc
            function(p)
                τ_troposphere  = p[22]
                CH₄_natural    = p[23]
                τ_soil         = p[24]
                τ_stratosphere = p[25]
                return logpdf(prior_τ_troposphere, τ_troposphere) + logpdf(prior_CH₄_natural, CH₄_natural) + logpdf(prior_τ_soil, τ_soil) + logpdf(prior_τ_stratosphere, τ_stratosphere)
            end

        # Throw an error for incorrect model selection.
        else
            error("Incorrect climate model selected. Options include :sneasy_fair, :sneasy_fund, :sneasy_hector, or :sneasy_magicc.")
        end

    #------------------------------------------------------------------------------------
    # Create function that returns the log-prior of all uncertain model parameters.
    #------------------------------------------------------------------------------------

    function total_log_prior_wider(p::Array{Float64,1})

        # Assign parameter values (common to all four models) names for convenience.
        σ_temperature      = p[1]
        σ_ocean_heat       = p[2]
        σ²_white_noise_CO₂ = p[3]
        σ²_white_noise_CH₄ = p[4]
        ρ_temperature      = p[5]
        ρ_ocean_heat       = p[6]
        α₀_CO₂             = p[7]
        α₀_CH₄             = p[8]
        temperature_0      = p[9]
        ocean_heat_0       = p[10]
        CO₂_0              = p[11]
        CH₄_0              = p[12]
        N₂O_0              = p[13]
        ECS                = p[14]
        heat_diffusivity   = p[15]
        rf_scale_aerosol   = p[16]
        rf_scale_CH₄       = p[17]
        F2x_CO₂            = p[18]
        Q10                = p[19]
        CO₂_fertilization  = p[20]
        CO₂_diffusivity    = p[21]

        # Return total log-prior of all uncertain parameters (using "ch4_model_priors" function to account for different versions of SNEASY+CH4).
        wider_log_prior = logpdf(prior_σ_temperature, σ_temperature) + logpdf(prior_σ_ocean_heat, σ_ocean_heat) + logpdf(prior_σ²_white_noise_CO₂, σ²_white_noise_CO₂) + logpdf(prior_σ²_white_noise_CH₄, σ²_white_noise_CH₄) +
                          logpdf(prior_ρ_temperature, ρ_temperature) + logpdf(prior_ρ_ocean_heat, ρ_ocean_heat) + logpdf(prior_α₀_CO₂, α₀_CO₂) + logpdf(prior_α₀_CH₄, α₀_CH₄) +
                          logpdf(prior_temperature_0, temperature_0) + logpdf(prior_ocean_heat_0, ocean_heat_0) + logpdf(prior_CO₂_0, CO₂_0) + logpdf(prior_CH₄_0, CH₄_0) + logpdf(prior_N₂O_0, N₂O_0) +
                          logpdf(prior_ECS, ECS) + logpdf(prior_heat_diffusivity, heat_diffusivity) + logpdf(prior_rf_scale_aerosol, rf_scale_aerosol) + logpdf(prior_rf_scale_CH₄, rf_scale_CH₄) + logpdf(prior_F2x_CO₂, F2x_CO₂) +
                          logpdf(prior_Q10, Q10) + logpdf(prior_CO₂_fertilization, CO₂_fertilization) + logpdf(prior_CO₂_diffusivity, CO₂_diffusivity) +
                          ch4_model_priors(p)

        return wider_log_prior
    end

    # Return total log-prior function for specified version of SNEASY+CH4.
    return total_log_prior_wider
end



#######################################################################################################################
# CALCULATE LOG POSTERIOR.
#######################################################################################################################
# Description: This creates a function that calculates the log-posterior probability of the uncertain model, initial
#              condition, and statistical process parameters specific to the CH₄ cycle model selected.
#
# Function Arguments:
#
#       f_run_model   = A function that runs the specific version of SNEASY+CH4 and returns the output being calibrated to observations.
#       climate_model = A symbol identifying the specific version of SNEASY+CH4 (options are :sneasy_fair, :sneasy_fund,
#                       :sneasy_hector, and :sneasy_magicc).
#       end_year      = The final year to run the model calibration (defaults to 2017).
#----------------------------------------------------------------------------------------------------------------------

function construct_log_posterior_wider(f_run_model, climate_model::Symbol; end_year::Int=2017)

    # Initialize model to start in 1765.
    start_year = 1765

    # Get log-prior function based on version of SNEASY+CH4 being used.
    total_log_prior_wider = construct_log_prior_wider(climate_model)

    # Create a vector of calibration years and calculate total number of years to run model.
    calibration_years = collect(start_year:end_year)
    n = length(calibration_years)

    # Load calibration data/observations.
    calibration_data = load_calibration_data(end_year)

    # Calculate indices for each year that has an observation in calibration data sets.
    indices_temperature_data   = findall(x-> !ismissing(x), calibration_data.hadcrut_temperature_obs)
    indices_oceanheat_data     = findall(x-> !ismissing(x), calibration_data.ocean_heat_obs)
    indices_oceanco2_flux_data = findall(x-> !ismissing(x), calibration_data.oceanco2_flux_obs)
    indices_maunaloa_co2_data  = findall(x-> !ismissing(x), calibration_data.maunaloa_co2_obs)
    indices_lawdome_co2_data   = findall(x-> !ismissing(x), calibration_data.lawdome_co2_obs)
    indices_noaa_ch4_data      = findall(x-> !ismissing(x), calibration_data.noaa_ch4_obs)
    indices_lawdome_ch4_data   = findall(x-> !ismissing(x), calibration_data.lawdome_ch4_obs)

    # Combine CO₂ and CH₄ indices from Law Dome and more recent observations.
    indices_co2_data = sort(vcat(indices_lawdome_co2_data, indices_maunaloa_co2_data))
    indices_ch4_data = sort(vcat(indices_noaa_ch4_data, indices_lawdome_ch4_data))

    # Combine CO₂ and CH₄ measurement errors from Law Dome and more recent observations (just for convenience).
    calibration_data.co2_combined_sigma = calibration_data.lawdome_co2_sigma
    calibration_data.co2_combined_sigma[indices_maunaloa_co2_data] = calibration_data.maunaloa_co2_sigma[indices_maunaloa_co2_data]

    calibration_data.ch4_combined_sigma = calibration_data.lawdome_ch4_sigma
    calibration_data.ch4_combined_sigma[indices_noaa_ch4_data] = calibration_data.noaa_ch4_sigma[indices_noaa_ch4_data]

    # Calculate number of ice core observations for CO₂ and CH₄ (used for indexing).
    n_lawdome_co2 = length(indices_lawdome_co2_data)
    n_lawdome_ch4 = length(indices_lawdome_ch4_data)

    # Allocate arrays to store data-model residuals.
    temperature_residual = zeros(length(indices_temperature_data))
    ocean_heat_residual  = zeros(length(indices_oceanheat_data))
    co2_residual         = zeros(length(indices_co2_data))
    ch4_residual         = zeros(length(indices_ch4_data))

    # Allocate array to store likelihoods for individual ocean CO₂ flux data points that need to be summed up to get a total likelihood (assuming iid error structure).
    oceanco2_flux_single_llik   = zeros(length(indices_oceanco2_flux_data))

    # Allocate vectors to store model output being calibrated to the observations.
    modeled_CO₂           = zeros(n)
    modeled_CH₄           = zeros(n)
    modeled_oceanCO₂_flux = zeros(n)
    modeled_temperature   = zeros(n)
    modeled_ocean_heat    = zeros(n)


    #---------------------------------------------------------------------------------------------------------------------------------------
    # Create a function to calculate the log-likelihood for the observations, assuming residual independence across calibration data sets.
    #---------------------------------------------------------------------------------------------------------------------------------------

    function total_log_likelihood_wider(p::Array{Float64,1})

        # Assign names to uncertain statistical process parameters used in log-likelihood calculations.
        σ_temperature      = p[1]
        σ_ocean_heat       = p[2]
        σ²_white_noise_CO₂ = p[3]
        σ²_white_noise_CH₄ = p[4]
        ρ_temperature      = p[5]
        ρ_ocean_heat       = p[6]
        α₀_CO₂             = p[7]
        α₀_CH₄             = p[8]

        # Run an instance of SNEASY+CH4 with sampled parameter set and return model output being compared to observations.
        f_run_model(p, modeled_CO₂, modeled_CH₄, modeled_oceanCO₂_flux, modeled_temperature, modeled_ocean_heat)


        #---------------------------------------------------------------------------
        # Global Surface Temperature (normalized to 1861-1880 mean) Log-Likelihood.
        #---------------------------------------------------------------------------

        llik_temperature = 0.0

        # Calculate temperature residuals.
        for (i, index)=enumerate(indices_temperature_data)
            temperature_residual[i] = calibration_data[index, :hadcrut_temperature_obs] - modeled_temperature[index]
        end

        # Calculate temperature log-likelihood.
        llik_temperature = hetero_logl_ar1(temperature_residual, σ_temperature, ρ_temperature, calibration_data[indices_temperature_data,:hadcrut_temperature_sigma])


        #-----------------------------------------------------------------------
        # Ocean Heat Content Log-Likelihood
        #-----------------------------------------------------------------------

        llik_ocean_heat = 0.0

        # Calculate ocean heat residuals.
        for (i, index)=enumerate(indices_oceanheat_data)
            ocean_heat_residual[i] = calibration_data[index, :ocean_heat_obs] - modeled_ocean_heat[index]
        end

        # Calculate ocean heat log-likelihood.
        llik_ocean_heat = hetero_logl_ar1(ocean_heat_residual, σ_ocean_heat, ρ_ocean_heat, calibration_data[indices_oceanheat_data, :ocean_heat_sigma])


        #-----------------------------------------------------------------------
        # Atmospheric CO₂ Concentration Log-Likelihood
        #-----------------------------------------------------------------------

        llik_co2 = 0.

        # Calculate CO₂ concentration (Law Dome) residuals (assuming 8 year model mean centered on year of ice core observation).
        for (i, index)=enumerate(indices_lawdome_co2_data)
            co2_residual[i] = calibration_data[index, :lawdome_co2_obs] - mean(modeled_CO₂[index .+ (-4:3)])
        end

        # Calculate CO₂ concentration (Mauna Loa) residuals.
        for (i, index)=enumerate(indices_maunaloa_co2_data)
            co2_residual[i+n_lawdome_co2] = calibration_data[index, :maunaloa_co2_obs] - modeled_CO₂[index]
        end

        # Calculate atmospheric CO₂ concentration log-likelihood.
        llik_co2 = hetero_logl_car1(co2_residual, indices_co2_data, σ²_white_noise_CO₂, α₀_CO₂, calibration_data[indices_co2_data, :co2_combined_sigma])


        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration Log-Likelihood
        #-----------------------------------------------------------------------

        llik_ch4 = 0.

        # Calculate CH₄ concentration (Law Dome) residuals for individual data points (assuming 8 year model mean centered on year of ice core observation).
        for (i,index) = enumerate(indices_lawdome_ch4_data)
            ch4_residual[i] = calibration_data[index, :lawdome_ch4_obs] - mean(modeled_CH₄[index .+ (-4:3)])
        end

        # Calculate CH₄ concentration (NOAA) residuals.
        for (i, index)=enumerate(indices_noaa_ch4_data)
            ch4_residual[i+n_lawdome_ch4] = calibration_data[index, :noaa_ch4_obs] - modeled_CH₄[index]
        end

        # Calculate atmospheric CH₄ concentration log-likelihood.
        llik_ch4 = hetero_logl_car1(ch4_residual, indices_ch4_data, σ²_white_noise_CH₄, α₀_CH₄, calibration_data[indices_ch4_data, :ch4_combined_sigma])


        #-----------------------------------------------------------------------
        # Ocean Carbon Flux Log-Likelihood
        #-----------------------------------------------------------------------

        llik_oceanco2_flux = 0.

        # Calculate ocean CO₂ flux log-likelihood for individual data points.
        for (i,index) = enumerate(indices_oceanco2_flux_data)
            oceanco2_flux_single_llik[i] = logpdf(Normal(modeled_oceanCO₂_flux[index], calibration_data[index, :oceanco2_flux_sigma]), calibration_data[index, :oceanco2_flux_obs])
        end

        # Calculate ocean CO₂ flux total log-likelihood as sum of individual data point likelihoods.
        llik_oceanco2_flux = sum(oceanco2_flux_single_llik)


        #-----------------------------------------------------------------------
        # Total Log-Likelihood
        #-----------------------------------------------------------------------

        # Calculate the total log-likelihood (assuming residual independence across data sets).
        llik = llik_temperature + llik_ocean_heat + llik_co2 + llik_ch4 + llik_oceanco2_flux

        return llik
    end


    #---------------------------------------------------------------------------------------------------------------
    # Create a function to calculate the log-posterior of uncertain parameters 'p' (posterior ∝ likelihood * prior)
    #---------------------------------------------------------------------------------------------------------------

    function log_posterior_wider(p)

        # Calculate log-prior
        log_prior_wider = total_log_prior_wider(p)

        # In case a parameter sample leads to non-physical model outcomes, return -Inf rather than erroring out.
        try
            log_post_wider = isfinite(log_prior_wider) ? total_log_likelihood_wider(p) + log_prior_wider : -Inf
        catch
            log_post_wider = - Inf
        end
    end

    # Return log posterior function given user specifications.
    return log_posterior_wider
end
