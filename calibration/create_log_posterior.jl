
#fix sort (should be sort!())

#RUN THIS THEN CHANGE INITIAL CONDITIONS OF CH4 IID ICE

#######################################################################################################################
# CALCULATE AR(1) LOG-LIKELIHOOD
########################################################################################################################
# Description: This function calculates the AR(1) log-likelihood in terms of the data-model residuls when accounting for
#              time-varying observation errors. It follows "The Effects of Time-Varying Observation Errors on Semi-Empirical
#              Sea-Level Projections" (Ruckert et al., 2017) DOI 10.1007/s10584-016-1858-z.
#
# Function Arguments:
#
#       residuals: A vector of data-model residuals.
#       σ:         AR(1) innovation standard deviation.
#       ρ:         AR(1) autocorrelation term.
#       ϵ:         A vector of time-varying observation error estimates (from calibration data sets).
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



# Create a function to calculate total (log) prior for model/statistical parameters (spefiic to model being calibrated)
function construct_log_prior(climate_model::Symbol)

    #Decalre all prior distributions for all uncertain parameters found in the four versions of SNEASY+CH4.
    # -----------------------------------------
    # Statistical Process Priors.
    # -----------------------------------------
    prior_σ_temperature     = Uniform(0, 0.2)
    prior_σ_ocean_heat      = Uniform(0, 4)
    prior_σ_CO₂inst         = Uniform(0, 1)
    prior_σ_CO₂ice          = Uniform(0, 10)
    prior_σ_CH₄inst         = Uniform(0, 20)
    prior_σ_CH₄ice          = Uniform(0, 45)

    prior_ρ_temperature     = Uniform(0, 0.99)
    prior_ρ_ocean_heat      = Uniform(0, 0.99)
    prior_ρ_CO₂inst         = Uniform(0, 0.99)
    prior_ρ_CH₄inst         = Uniform(0, 0.99)
    prior_ρ_CH₄ice          = Uniform(0, 0.99)

    # -----------------------------------------
    # Initial Condition Priors.
    # -----------------------------------------
    prior_temperature_0     = Normal()
    prior_ocean_heat_0      = Uniform(-100, 0)
    prior_CO₂_0             = Uniform(275, 281)
    prior_CH₄_0             = Uniform(691, 753)
    prior_N₂O_0             = Uniform(264, 282)

    # -----------------------------------------
    # Climate & Radiative Forcing Priors.
    # -----------------------------------------
    prior_ECS               = Truncated(Cauchy(3.0,2.0), 0.0, 10.0)
    prior_heat_diffusivity  = LogNormal(1.1, 0.3)
    prior_rf_scale_aerosol  = TriangularDist(0., 3., 1.)
    prior_rf_scale_CH₄      = Uniform(0.72, 1.28)
    prior_F2x_CO₂           = Uniform(2.968, 4.452)

    # -----------------------------------------
    # Carbon Cycle Priors.
    # -----------------------------------------
    prior_Q10               = Uniform(1.0, 5)
    prior_CO₂_fertilization = Uniform(0., 1)
    prior_CO₂_diffusivity   = Uniform(0., 200)

    # -----------------------------------------
    # Methane Cycle Priors.
    # -----------------------------------------
    prior_τ_troposphere     = Uniform(6.7, 13.5)
    prior_CH₄_natural       = Uniform(150, 484)
    prior_τ_soil            = Uniform(96, 144)
    prior_τ_stratosphere    = Uniform(100, 200)


    #----------------------------------------------------------------------------------------------------------------
    # Create a function unique to each CH₄ cycle model to calculate prior distribution for uncertain CH₄ parameters.
    #----------------------------------------------------------------------------------------------------------------

    # CH₄ prior function.
    ch4_model_priors =

        # Create function based on uncertain SNEASY-FAIR CH₄ cycle parameters.
        if climate_model == :sneasy_fair
            function(p)
                τ_troposphere = p[25]
                CH₄_natural   = p[26]
                return logpdf(prior_τ_troposphere, τ_troposphere) + logpdf(prior_CH₄_natural, CH₄_natural)
            end

        # Create function based on uncertain SNEASY-FUND CH₄ cycle parameters (note, FUND does not have extra uncertain parameters).
        elseif climate_model == :sneasy_fund
            function(p)
                τ_troposphere = p[25]
                return logpdf(prior_τ_troposphere, τ_troposphere)
            end

        # Create function based on uncertain SNEASY-Hector or SNEASY-MAGICC CH₄ cycle parameters.
        elseif climate_model == :sneasy_hector || climate_model == :sneasy_magicc
            function(p)
                τ_troposphere  = p[25]
                CH₄_natural    = p[26]
                τ_soil         = p[27]
                τ_stratosphere = p[28]
                return logpdf(prior_τ_troposphere, τ_troposphere) + logpdf(prior_CH₄_natural, CH₄_natural) + logpdf(prior_τ_soil, τ_soil) + logpdf(prior_τ_stratosphere, τ_stratosphere)
            end

        # Throw an error for incorrect model selection.
        else
            error("Incorrect climate model selected. Options include :sneasy_fair, :sneasy_fund, :sneasy_hector, or :sneasy_magicc.")
        end


    #------------------------------------------------------------------------------------
    # Create function that returns the log-prior of all uncertain model parameters.
    #------------------------------------------------------------------------------------

    function total_log_prior(p::Array{Float64,1})

        # Assign parameter values (common to all four models) names for convenience.
        σ_temperature     = p[1]
        σ_ocean_heat      = p[2]
        σ_CO₂inst         = p[3]
        σ_CO₂ice          = p[4]
        σ_CH₄inst         = p[5]
        σ_CH₄ice          = p[6]
        ρ_temperature     = p[7]
        ρ_ocean_heat      = p[8]
        ρ_CO₂inst         = p[9]
        ρ_CH₄inst         = p[10]
        ρ_CH₄ice          = p[11]
        temperature_0     = p[12]
        ocean_heat_0      = p[13]
        CO₂_0             = p[14]
        CH₄_0             = p[15]
        N₂O_0             = p[16]
        ECS               = p[17]
        heat_diffusivity  = p[18]
        rf_scale_aerosol  = p[19]
        rf_scale_CH₄      = p[20]
        F2x_CO₂           = p[21]
        Q10               = p[22]
        CO₂_fertilization = p[23]
        CO₂_diffusivity   = p[24]

        # Return total log-prior of all uncertain parameters.
        log_prior = logpdf(prior_σ_temperature, σ_temperature) + logpdf(prior_σ_ocean_heat, σ_ocean_heat) + logpdf(prior_σ_CO₂inst, σ_CO₂inst) + logpdf(prior_σ_CO₂ice, σ_CO₂ice) + logpdf(prior_σ_CH₄inst, σ_CH₄inst) + logpdf(prior_σ_CH₄ice, σ_CH₄ice) +
                    logpdf(prior_ρ_temperature, ρ_temperature) + logpdf(prior_ρ_ocean_heat, ρ_ocean_heat) + logpdf(prior_ρ_CO₂inst, ρ_CO₂inst) + logpdf(prior_ρ_CH₄inst, ρ_CH₄inst) + logpdf(prior_ρ_CH₄ice, ρ_CH₄ice) +
                    logpdf(prior_temperature_0, temperature_0) + logpdf(prior_ocean_heat_0, ocean_heat_0) + logpdf(prior_CO₂_0, CO₂_0) + logpdf(prior_CH₄_0, CH₄_0) + logpdf(prior_N₂O_0, N₂O_0) +
                    logpdf(prior_ECS, ECS) + logpdf(prior_heat_diffusivity, heat_diffusivity) + logpdf(prior_rf_scale_aerosol, rf_scale_aerosol) + logpdf(prior_rf_scale_CH₄, rf_scale_CH₄) + logpdf(prior_F2x_CO₂, F2x_CO₂) +
                    logpdf(prior_Q10, Q10) + logpdf(prior_CO₂_fertilization, CO₂_fertilization) + logpdf(prior_CO₂_diffusivity, CO₂_diffusivity) +
                    ch4_model_priors(p)

        return log_prior
    end

    # Return total log-prior function for specified version of SNEASY+CH4.
    return total_log_prior
end


function construct_log_posterior(f_run_model, climate_model::Symbol; end_year::Int=2017)

    # Initialize models in 1765.
    start_year = 1765

    # Get log-prior function based on version of SNEASY+CH4 being used.
    total_log_prior = construct_log_prior(climate_model)

    # Create a vector of calibration years and parameter for number of years.
    calibration_years = collect(start_year:end_year)
    n = length(calibration_years)

    # Load calibration data/observations.
    calibration_data = load_calibration_data(start_year, end_year)

	# Calculate indices for each year that has an observation in calibration data sets.
    indices_temperature_data   = findall(x-> !ismissing(x), calibration_data.hadcrut_temperature_obs)
    indices_oceanheat_data     = findall(x-> !ismissing(x), calibration_data.ocean_heat_obs)
    indices_oceanco2_flux_data = findall(x-> !ismissing(x), calibration_data.oceanco2_flux_obs)
    indices_maunaloa_co2_data  = findall(x-> !ismissing(x), calibration_data.maunaloa_co2_obs)
    indices_lawdome_co2_data   = findall(x-> !ismissing(x), calibration_data.lawdome_co2_obs)
    indices_noaa_ch4_data      = findall(x-> !ismissing(x), calibration_data.noaa_ch4_obs)

    # Calculate indices for i.i.d. and AR(1) blocks in CH₄ ice core data.
    lawdome_ch4_ar1_start_index, lawdome_ch4_ar1_end_index, lawdome_ch4_ar1_indices, lawdome_ch4_iid_indices = ch4_indices(calibration_data.lawdome_ch4_obs)

    # Allocate arrays to calculate data-model residuals.
    temperature_residual     = zeros(length(indices_temperature_data))
    ocean_heat_residual      = zeros(length(indices_oceanheat_data))
    maunaloa_co2_residual    = zeros(length(indices_maunaloa_co2_data))
    noaa_ch4_residual        = zeros(length(indices_noaa_ch4_data))
    lawdome_ch4_ar1_residual = zeros(length(lawdome_ch4_ar1_indices))

    # Allocate arrays for iid ice core data to store 8 year means of model output.
    lawdome_co2_mean          = zeros(length(indices_lawdome_co2_data))
    lawdome_ch4_mean          = zeros(length(lawdome_ch4_iid_indices))

    # Allocate arrays to store likelihoods for individual data points that need to be summed up to get a total likelihood.
    lawdome_co2_single_llik     = zeros(length(indices_lawdome_co2_data))
    oceanco2_flux_single_llik   = zeros(length(indices_oceanco2_flux_data))
    lawdome_ch4_single_llik_iid = zeros(length(lawdome_ch4_iid_indices))
    lawdome_ch4_block_llik_ar1  = zeros(length(lawdome_ch4_ar1_start_index))

	# Allocate vectors to store model output being compared to the observations.
	modeled_CO₂           = zeros(n)
	modeled_CH₄           = zeros(n)
	modeled_oceanCO₂_flux = zeros(n)
	modeled_temperature   = zeros(n)
	modeled_ocean_heat    = zeros(n)

    #---------------------------------------------------------------------------------------------------------------------------------------
    # Create a function to calculate the log-likelihood for the observations, assuming residual independence across calibration data sets.
    #---------------------------------------------------------------------------------------------------------------------------------------

	function total_log_likelihood(p::Array{Float64,1})

        # Assign names to uncertain statistical process parameters used in log-likelihood calculations.
        σ_temperature     = p[1]
        σ_ocean_heat      = p[2]
        σ_CO₂inst         = p[3]
        σ_CO₂ice          = p[4]
        σ_CH₄inst         = p[5]
        σ_CH₄ice          = p[6]
        ρ_temperature     = p[7]
        ρ_ocean_heat      = p[8]
        ρ_CO₂inst         = p[9]
        ρ_CH₄inst         = p[10]
        ρ_CH₄ice          = p[11]

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
        # Atmospheric CO₂ Concentration Log-Likelihood (Mauna Loa)
        #-----------------------------------------------------------------------

        llik_maunaloa_co2 = 0.

        # Calculate CO₂ concentration (Mauna Loa) residuals.
        for (i, index)=enumerate(indices_maunaloa_co2_data)
            maunaloa_co2_residual[i] = calibration_data[index, :maunaloa_co2_obs] - modeled_CO₂[index]
        end

        # Calculate CO₂ concentration (Mauna Loa) log-likelihood.
        llik_maunaloa_co2 = hetero_logl_ar1(maunaloa_co2_residual, σ_CO₂inst, ρ_CO₂inst, calibration_data[indices_maunaloa_co2_data, :maunaloa_co2_sigma])


        #-----------------------------------------------------------------------
        # Atmospheric CO₂ Concentration Log-Likelihood (Law Dome)
        #-----------------------------------------------------------------------

        llik_lawdome_co2 = 0.

        # Calculate CO₂ concentration (Law Dome) log-likelihoods for individual data points (assuming 8 year model mean centered on year of ice core observation.
        for (i, index)=enumerate(indices_lawdome_co2_data)
            lawdome_co2_mean[i] = mean(modeled_CO₂[index .+ (-4:3)])
            lawdome_co2_single_llik[i] = logpdf(Normal(lawdome_co2_mean[i], sqrt(σ_CO₂ice^2 + calibration_data[index, :lawdome_co2_sigma]^2)), calibration_data[index, :lawdome_co2_obs])
        end

        # Calculate CO₂ concentration (Law Dome) total log-likelihood as sum of individual data point likelihoods.
        llik_lawdome_co2 = sum(lawdome_co2_single_llik)


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
        # Atmospheric CH₄ Concentration Log-Likelihood (NOAA)
        #-----------------------------------------------------------------------

        llik_noaa_ch4 = 0.

        # Calculate CH₄ concentration (NOAA) residuals.
        for (i, index)=enumerate(indices_noaa_ch4_data)
            noaa_ch4_residual[i] = calibration_data[index, :noaa_ch4_obs] - modeled_CH₄[index]
        end

        # Calculate CH₄ concentration (NOAA) log-likelihood.
        llik_noaa_ch4 = hetero_logl_ar1(noaa_ch4_residual, σ_CH₄inst, ρ_CH₄inst, calibration_data[indices_noaa_ch4_data, :noaa_ch4_sigma])

        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration i.i.d Blocks Log-Likelihood (Law Dome)
        #-----------------------------------------------------------------------

        llik_lawdome_ch4_iid = 0.

        # Calculate CH₄ concentration (Law Dome) log-likelihoods for individual data points (assuming 8 year model mean centered on year of ice core observation).
        for (i,index) = enumerate(lawdome_ch4_iid_indices)
            lawdome_ch4_mean[i] = mean(modeled_CH₄[index .+ (-4:3)])
            lawdome_ch4_single_llik_iid[i] = logpdf(Normal(lawdome_ch4_mean[i], sqrt(σ_CH₄ice^2 + calibration_data[index, :lawdome_ch4_sigma]^2)), calibration_data[index, :lawdome_ch4_obs])
        end

        # Calculate CH₄ concentration (Law Dome) total log-likelihood as sum of individual data point likelihoods.
        llik_lawdome_ch4_iid = sum(lawdome_ch4_single_llik_iid)

        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration AR1 Blocks Log-Likelihood (Law Dome)
        #-----------------------------------------------------------------------

        llik_lawdome_ch4_ar1 = 0.

        # Use counter to track residuals for various blocks (for convenience).
        let
        counter =0

            # Loop through every AR(1) block.
            for block = 1:length(lawdome_ch4_ar1_start_index)
                # Within a single AR1 block, calculate residuals (8 year model average, centered on year of ice core observation).
                for index in collect(lawdome_ch4_ar1_start_index[block]:lawdome_ch4_ar1_end_index[block])
                    counter+=1
                    lawdome_ch4_ar1_residual[counter] = calibration_data[index, :lawdome_ch4_obs] - mean(modeled_CH₄[index .+ (-4:3)])
                end
                # Calculate log likelihood for that block and add to total likelihood for ch4 AR(1) ice core data. (Do indexing to avoid having to allocate a new residual array for every block).
                lawdome_ch4_block_llik_ar1[block] = hetero_logl_ar1(lawdome_ch4_ar1_residual[(counter-length(lawdome_ch4_ar1_start_index[block]:lawdome_ch4_ar1_end_index[block])+1):counter], σ_CH₄ice, ρ_CH₄ice, calibration_data[lawdome_ch4_ar1_start_index[block]:lawdome_ch4_ar1_end_index[block], :lawdome_ch4_sigma])
            end
        end

        llik_lawdome_ch4_ar1 = sum(lawdome_ch4_block_llik_ar1)


        #-----------------------------------------------------------------------
        # Total Log-Likelihood
        #-----------------------------------------------------------------------

        # Calculate the total log-likelihood (assuming residual independence across data sets).
        llik = llik_temperature + llik_ocean_heat + llik_maunaloa_co2 + llik_lawdome_co2 + llik_oceanco2_flux + llik_noaa_ch4 + llik_lawdome_ch4_iid + llik_lawdome_ch4_ar1

		return llik
	end

	# (log) posterior distribution:  posterior ~ likelihood * prior
	function log_posterior(p)
		log_prior = total_log_prior(p)
		# In case a parameter sample leads to non-physical model outcomes, return -Inf rather than erroring out.
        try
            log_post = isfinite(log_prior) ? total_log_likelihood(p) + log_prior : -Inf
        catch
            log_post = - Inf
        end
	end

	return log_posterior
end
