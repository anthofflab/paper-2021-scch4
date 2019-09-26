using Distributions
using DataFrames
using DataFramesMeta
using CSV 

include(joinpath(dirname(@__FILE__), "..", "calibration_helper_functions.jl"))
#include(joinpath(dirname(@__FILE__), "..", "..", "..", "sneasy", "julia", "sneasy.jl"))
include("run_mimi_sneasy.jl")

#=
function sneasy_load_data()
	# load emissions data time series.
    #Foss_Hist = readtable("../../src/Sneasy_magicc/FOSS_HIST_modified_timestep.csv", separator = ',', header=true)
    Foss_Hist = readtable(joinpath(dirname(@__FILE__), "../data/FOSS_HIST_modified_timestep.csv"), separator = ',', header=true)
    Foss_Hist = DataFrame(year=Foss_Hist[:YEAR], EFOSS=Foss_Hist[:EFOSS])

    rf_ch4_data = readtable(joinpath(dirname(@__FILE__), "..","data","RCP85_MIDYEAR_RADFORCING.csv"), separator = ',', header=true);
    rf_ch4_data = DataFrame(year=rf_ch4_data[:YEARS], CH4_RF=rf_ch4_data[:CH4_RF], ch4_h20_rf = rf_ch4_data[:CH4OXSTRATH2O_RF], TROPOZ_RF = rf_ch4_data[:TROPOZ_RF])

    df_emissions = readtable(joinpath(dirname(@__FILE__),"../data/RCP85_EMISSIONS.csv"))
	rename!(df_emissions, :YEARS, :year)

	df_forcings = readtable(joinpath(dirname(@__FILE__),"../data/forcing_rcp85.txt"), separator=' ')

	#df = join(df_emissions, df_forcings, on=:year, kind=:outer)
    #df = join(df, rf_ch4_data, on=:year, kind=:outer)
    #df = join(df, Foss_Hist, on=:year, kind=:outer)

    df = join(Foss_Hist, df_emissions, on=:year, kind=:outer)
    df = join(df, rf_ch4_data, on=:year, kind=:outer)
    df = join(df, df_forcings, on=:year, kind=:outer)
   # df = DataFrame(year=df[:year], co2=df[:FossilCO2]+df[:OtherCO2], ch4_emissions=df[:CH4], rf_aerosols=df[:aerosol_direct]+df[:aerosol_indirect], rf_other=df[:ghg_nonco2]+df[:solar]+df[:volcanic]+df[:other]-df[:ch4_rf])
    df = DataFrame(year=df[:year], co2=df[:FossilCO2]+df[:OtherCO2], ch4_emissions=df[:CH4], nox_emissions=df[:NOx], co_emissions=df[:CO], nmvoc_emissions=df[:NMVOC], foss_hist=df[:EFOSS], rf_aerosols=df[:aerosol_direct]+df[:aerosol_indirect], rf_other=df[:ghg_nonco2]+df[:volcanic]+df[:solar]+df[:other]- df[:CH4_RF] - df[:ch4_h20_rf] - df[:TROPOZ_RF])

	return df
end
=#

#EXACT AR1 FROM TONY's CODE???
function hetero_logl_ar1(r, σ, ρ, ϵ)
    n=length(r)
    σ_process = σ/sqrt(1-ρ^2)

    #Check if all observation errors are 0
#=    # if all(x -> x==0, ϵ)
        # Calculate log-likelihood for first value.
        logl = logpdf(Normal(0, σ_process), r[1])
        # Calulate total likelihood for the 2:n data-model residuals.
        for i =2:n
            #Whiten each residual.
            w = r[i] - ρ * r[i-1]
            logl = logl + logpdf(Normal(0, sqrt(σ^2 + ϵ[i]^2)), w)
        end
=#
    # Else if all observation errors are not 0...
 #   else
        # Replicate outer-product function in R code.
        H = abs.(collect(1:n)' .- collect(1:n))
        # Covariance matrix for residual component consisting of stationary normal AR(1) first-order autoregressive process.
        v = σ_process ^2 * ρ .^ H
        # Add variance of observation errors to diagonal of AR(1) covariance matrix.
    #    if length(ϵ) > 1
        v = v + diagm(ϵ.^2)
     #   else
            # Replicate observation error value for all observations if only one error value is given.
     #       v = v + diagm(ones(n).*ϵ.^2)
     #   end
        # Calculate log likelihood.
        logl = logpdf(MvNormal(v), r)
    #end
    return logl
end


# log likelihood for a zero-mean AR1 process (innovation variance sigma^2, lag-1 autocorrelation coefficient rho1); handles initial value assuming process stationarity
function loglar1(r, σ, ρ1)
	n = length(r)
	σ_proc = σ/sqrt(1-ρ1^2) # stationary process variance sigma.proc^2

  	logl = logpdf(Normal(0, σ_proc), r[1])

  	for i=2:length(r)
		w = r[i] - ρ1 * r[i-1] # whitened residuals
		logl = logl + logpdf(Normal(0, σ), w)
	end
  	return logl
end

# log likelihood function that switches statistical parameters estimated from ch4 ice core to flask observations across appropirate time periods.
function double_ar1_sim(N1, rho1, sigma1, N2, rho2, sigma2)
    x = zeros(N1+N2)
    x[1] = sigma1/sqrt(1-rho1^2)
    # Simulate noise for CH4 ice core period.
    for i in 2:N1
        x[i] = rho1 * x[i-1] + rand(Normal(0, sigma1))
    end
    # Simulate noise for CH4 flask ch4 measurements.
    for i in (N1+1):(N1+N2)
        x[i] = rho2 * x[i-1] + rand(Normal(0, sigma2))
    end
    return x
end


# (log) prior for model/statistical parameters
function log_pri(p)
	S = p[1]
	κ = p[2]
	α = p[3]
	Q10 = p[4]
	beta = p[5]
	eta = p[6]
	T0 = p[7]
	H0 = p[8]
	CO20 = p[9]
	σ_temp = p[10]
	σ_ocheat = p[11]
	σ_co2inst = p[12]
	σ_co2ice = p[13]
	ρ_temp = p[14]
	ρ_ocheat = p[15]
	ρ_co2inst = p[16]
    τ = p[17]
    σ_ch4inst = p[18]
    ρ_ch4inst = p[19]
    σ_ch4ice = p[20]
    ρ_ch4ice = p[21]
    CH4_0 = p[22]
    CH4_Nat = p[23]
    N2O_0 = p[24]
    F2x_CO₂ = p[25]
    rf_scale_CH₄ = p[26]

	#prior_S1 = NormalInverseGaussian(1.7, 1.8, 1.2, 2.3)
	#prior_S2 = NormalInverseGaussian(1.3, 1.9, 1.0, 3.3)
	prior_S = Truncated(Cauchy(3.0,2.0), 0.0, 10.0)
    prior_κ = LogNormal(1.1, 0.3)
	prior_α = TriangularDist(0., 3., 1.)
	prior_Q10 = Uniform(1.0, 5)
	prior_beta = Uniform(0., 1)
	prior_eta = Uniform(0., 200)
	prior_T0 = Normal()
	prior_H0 = Uniform(-100, 0)
	#prior_CO20 = Uniform(280, 295)
    # IPCC notes 1750 value of 278 ± 2 (which seems to be 90% CI).  278.05158
    #NOTE TO REMOVE: asumme normal dist, 90% ≈ 1.65 SDs.  Rescale to be ± 2SDs (backing this out gives ethridge 1σ values he reports)
    #prior_CO20 = Uniform(276, 281)
	prior_CO20 = Uniform(275, 281)
    prior_σ_temp = Uniform(0, 0.2)
	prior_σ_ocheat = Uniform(0, 4)
	prior_σ_co2inst = Uniform(0, 1)
	prior_σ_co2ice = Uniform(0, 10)
	prior_ρ_temp = Uniform(0, 0.99)
	prior_ρ_ocheat = Uniform(0, 0.99)
	prior_ρ_co2inst = Uniform(0, 0.99)
    #"Preindustrial to present-day changes in tropospheric hydroxyl radical and methane lifetime from the Atmospheric Chemistry and Climate Model Intercomparison Project (ACCMIP)"
    # From this paper, use multi-model mean 1850 lifetime ± 2 standard deviations (mmm ± 1SD = 10.1 ± 1.7 years)
    prior_τ = Uniform(6.7, 13.5)
    prior_σ_ch4inst = Uniform(0, 20)
    prior_ρ_ch4inst = Uniform(0, 0.99)

    # 1σ precision (air extraction and analysis) for ice core = 5ppb, 37% interhemispheric scaling introduces at most ≈ 10ppb error (Etheridge 1998). Doubled prior to not hit boundary for first pass.
    #prior_σ_ch4ice = Uniform(0, 30)
    prior_σ_ch4ice = Uniform(0, 45)

    prior_ρ_ch4ice = Uniform(0, 0.99)
    # RCP 1765 CH₄ concentration range spans max prior of 1σ for ch4ice obs, centered on 1765 RCP value.
    # IPCC AR5 WG1 Table2.1 Notes 1750 CH4 from ice core ≈ 722 ± 25ppb. Using 25 as range of 1850 RCP value.
    #prior_CH4_0 = Uniform(765, 816)
    #prior_CH4_0 = Uniform(696, 747)
    prior_CH4_0 = Uniform(691, 753)

    #prior_CH4_2 = Uniform(697, 747)
    
    # USE THIS ONE!!!!!!!!!
    # From IPCC Chapter 6 Table 6.8... has table like Nature Geoscience of top down vs bottom up natural emission estimates
    # Lowest value = 150, highest = 484.
    prior_CH4_Nat = Uniform(150, 484)

    # From IPCC Natural wetland emissions = 177-284 Tg/yr, Other natural emissions = 61-200 Tg/yr
    # IPCC AR5 WG1 Chapt 6 Pg 467
    #prior_CH4_Nat = Uniform(238, 484)
    
    #Reactive greenhouse gas scenarios: Systematic explorationof uncertainties and the role of atmospheric chemistry Prather 2012
    # natural emissions = 202 ± 35 (where 35 = 1SD).  Using 2 SDs gives range of [132,272]
    #prior_CH4_Nat = Uniform(132, 272)

    # IPCC AR5 WG1 Table2.1 Notes 1750 N2O has uncertinaty of ± 7ppb. Using this as range, with 1765 RCP value (272.95961)
    # rescaling from 90% (about 1.65 SD to 2 SD 95%CI)
    prior_N2O_0 = Uniform(264, 282)
    #prior_N2O_0 = Uniform(268, 283)
    #prior_N2O_0 = Uniform(265, 280)
    
    # Prior for forcing from doubling of CO₂ (taken from FAIR 1.3 = ±20% best guess of 3.71)
    prior_F2x_CO₂ = Uniform(2.968, 4.452)
    # Prior for direct CH₄ forcing (taken from FAIR 1.3 = ±28%)
    prior_rf_scale_CH₄ = Uniform(0.72, 1.28)

	# For climate sensitivyt the R code uses a truncated distribution in the R
	# code with support 0 to Inf. The best way to fix this would be to add
	# support for a truncated NormalInverseGaussian to the Distributions package.
	# For now, the code below just checks the bounds manually, and then uses
	# the pdf of the NormalInverseGaussian without any truncation. Technically
	# that is not the correct pdf, but for the MCMC algorithm that doesn't matter.
	lpri = -Inf
	if S>0.
        #lpri = logpdf(prior_S1, S) + logpdf(prior_S2, S) + logpdf(prior_κ, κ) + logpdf(prior_α, α) + logpdf(prior_Q10, Q10) + logpdf(prior_beta, beta) + logpdf(prior_eta, eta) + logpdf(prior_T0, T0) + logpdf(prior_H0, H0) + logpdf(prior_CO20, CO20) + logpdf(prior_σ_temp, σ_temp) + logpdf(prior_σ_ocheat, σ_ocheat) + logpdf(prior_σ_co2inst, σ_co2inst) + logpdf(prior_σ_co2ice, σ_co2ice) + logpdf(prior_ρ_temp, ρ_temp) + logpdf(prior_ρ_ocheat, ρ_ocheat) + logpdf(prior_ρ_co2inst, ρ_co2inst) + logpdf(prior_τ, τ) +logpdf(prior_σ_ch4inst, σ_ch4inst) + logpdf(prior_ρ_ch4inst, ρ_ch4inst) +logpdf(prior_σ_ch4ice, σ_ch4ice) + logpdf(prior_ρ_ch4ice, ρ_ch4ice) + logpdf(prior_CH4_0, CH4_0) + logpdf(prior_CH4_Nat, CH4_Nat) + logpdf(prior_N2O_0, N2O_0) + logpdf(prior_F2x_CO₂, F2x_CO₂) + logpdf(prior_rf_scale_CH₄, rf_scale_CH₄)
        lpri = logpdf(prior_S, S) + logpdf(prior_κ, κ) + logpdf(prior_α, α) + logpdf(prior_Q10, Q10) + logpdf(prior_beta, beta) + logpdf(prior_eta, eta) + logpdf(prior_T0, T0) + logpdf(prior_H0, H0) + logpdf(prior_CO20, CO20) + logpdf(prior_σ_temp, σ_temp) + logpdf(prior_σ_ocheat, σ_ocheat) + logpdf(prior_σ_co2inst, σ_co2inst) + logpdf(prior_σ_co2ice, σ_co2ice) + logpdf(prior_ρ_temp, ρ_temp) + logpdf(prior_ρ_ocheat, ρ_ocheat) + logpdf(prior_ρ_co2inst, ρ_co2inst) + logpdf(prior_τ, τ) +logpdf(prior_σ_ch4inst, σ_ch4inst) + logpdf(prior_ρ_ch4inst, ρ_ch4inst) +logpdf(prior_σ_ch4ice, σ_ch4ice) + logpdf(prior_ρ_ch4ice, ρ_ch4ice) + logpdf(prior_CH4_0, CH4_0) + logpdf(prior_CH4_Nat, CH4_Nat) + logpdf(prior_N2O_0, N2O_0) + logpdf(prior_F2x_CO₂, F2x_CO₂) + logpdf(prior_rf_scale_CH₄, rf_scale_CH₄)
    end

	# This is just to emulate the R code right now, i.e. if the prior is finite
	# we are recalculating the prior to be just some parameters, see issue #9
	# TODO remove this whole if clause once things are cross checked
	if isfinite(lpri)
		#lpri = logpdf(prior_S1, S) + logpdf(prior_S2, S) + logpdf(prior_κ, κ) + logpdf(prior_α, α) + logpdf(prior_T0, T0)
        lpri = logpdf(prior_S, S) + logpdf(prior_κ, κ) + logpdf(prior_α, α) + logpdf(prior_T0, T0)
    end

	return lpri
end

function construct_log_post(f_run_model; start_year=1765, end_year=2017, assim_temp=true, assim_ocheat=true, assim_co2inst=true, assim_co2ice=true, assim_ocflux=true, assim_ch4inst = true, assim_ch4ice=true)
	
    calibration_years = collect(start_year:end_year)

    # Length of calibration period.
    n = length(start_year:end_year)

    # Load calibration data.
    calibration_data = load_calibration_data(start_year, end_year)

	# Calculate indices for each year that has an observation in calibration data sets.
    obs_temperature_indices  = find(x-> !ismissing(x), calibration_data[:hadcrut_temperature])
    obs_ocheat_indices       = find(x-> !ismissing(x), calibration_data[:obs_ocheat])
    obs_ocflux_indices       = find(x-> !ismissing(x), calibration_data[:obs_ocflux])
    obs_co2inst_indices      = find(x-> !ismissing(x), calibration_data[:obs_co2inst])
    obs_co2ice_indices       = find(x-> !ismissing(x), calibration_data[:obs_co2ice])
    obs_ch4inst_indices      = find(x-> !ismissing(x), calibration_data[:obs_ch4inst])

    # Calculate indices for i.i.d. and AR(1) blocks in CH₄ ice core data.
    ch4ice_ar1_start, ch4ice_ar1_end, ch4ice_ar1_indices, ch4ice_iid_indices = ch4_indices(calibration_data[:obs_ch4ice])

    # Allocate arrays to calculate data-model residuals.
    temperature_residual  = zeros(length(obs_temperature_indices))
    ocheat_residual       = zeros(length(obs_ocheat_indices))
    co2inst_residual      = zeros(length(obs_co2inst_indices))
    ch4inst_residual      = zeros(length(obs_ch4inst_indices))
    # Used to calculate 8 year mean, centered on year of CO2 ice observation.
    co2ice_mean           = zeros(length(obs_co2ice_indices))
    # Allocate arrays for residuals (AR1 blocks) and mean value (iid data).
    ch4ice_ar1_residual = zeros(length(ch4ice_ar1_indices))
    ch4ice_mean = zeros(length(ch4ice_iid_indices))

	# Allocate vectors for results
	model_co2 = zeros(n)
	model_ch4 = zeros(n)
	model_ocflux = zeros(n)
	model_temperature = zeros(n)
	model_ocheat = zeros(n)

	# (log) likelihood for observations, assuming residual independence between data sets
	function sneays_log_lik(p)
		S = p[1]
        κ = p[2]
        α = p[3]
        Q10 = p[4]
        beta = p[5]
        eta = p[6]
        T0 = p[7]
        H0 = p[8]
        CO20 = p[9]
        σ_temp = p[10]
        σ_ocheat = p[11]
        σ_co2inst = p[12]
        σ_co2ice = p[13]
        ρ_temp = p[14]
        ρ_ocheat = p[15]
        ρ_co2inst = p[16]
        τ = p[17]
        σ_ch4inst = p[18]
        ρ_ch4inst = p[19]
        σ_ch4ice = p[20]
        ρ_ch4ice = p[21]
        CH4_0 = p[22]
        CH4_Nat = p[23]
        N2O_0 = p[24]
        F2x_CO₂ = p[25]
        rf_scale_CH₄ = p[26]

		f_run_model(model_co2, model_ch4, model_ocflux, model_temperature, model_ocheat,
			S, κ, α, Q10, beta, eta, τ, CH4_Nat, CO20, T0, H0, CH4_0, N2O_0, F2x_CO₂, rf_scale_CH₄)

        #-----------------------------------------------------------------------
        # Temperature (normalized to 1850-1870 temp mean)
        #-----------------------------------------------------------------------
        llik_temp = 0.0
        if assim_temp
            for (i, index)=enumerate(obs_temperature_indices)
                temperature_residual[i] = calibration_data[index, :hadcrut_temperature] - model_temperature[index]
            end
            llik_temp = hetero_logl_ar1(temperature_residual, σ_temp, ρ_temp, calibration_data[obs_temperature_indices,:hadcrut_err])
        end

        #-----------------------------------------------------------------------
        # Ocean Heat content
        #-----------------------------------------------------------------------
        # TODO: Look into this note: (doesn't need to be normalized because H0 offsets this based on BRICK notes).
        llik_ocheat = 0.0
        if assim_ocheat
            for (i, index)=enumerate(obs_ocheat_indices)
                ocheat_residual[i] = calibration_data[index, :obs_ocheat] - model_ocheat[index]
            end
            llik_ocheat = hetero_logl_ar1(ocheat_residual, σ_ocheat, ρ_ocheat, calibration_data[obs_ocheat_indices, :obs_ocheatsigma])
        end

        #-----------------------------------------------------------------------
        # Atmospheric CO₂ Concentration (instrumental)
        #-----------------------------------------------------------------------
        llik_co2inst = 0.
        if assim_co2inst
            for (i, index)=enumerate(obs_co2inst_indices)
                co2inst_residual[i] = calibration_data[index, :obs_co2inst] - model_co2[index]
            end
            llik_co2inst = hetero_logl_ar1(co2inst_residual, σ_co2inst, ρ_co2inst, calibration_data[obs_co2inst_indices, :co2inst_sigma])
        end

        #-----------------------------------------------------------------------
        # Atmospheric CO₂ Concentration (ice core)
        #-----------------------------------------------------------------------
        llik_co2ice = 0.
        if assim_co2ice
            for (i, index)=enumerate(obs_co2ice_indices)
                co2ice_mean[i] = mean(model_co2[index + (-4:3)])
                llik_co2ice = llik_co2ice + logpdf(Normal(co2ice_mean[i], sqrt(σ_co2ice^2 + calibration_data[index, :co2ice_sigma]^2)), calibration_data[index, :obs_co2ice])
            end
        end

        #-----------------------------------------------------------------------
        # Ocean Carbon Flux
        #-----------------------------------------------------------------------
        llik_ocflux = 0.
        if assim_ocflux
            for (i,index) = enumerate(obs_ocflux_indices)
                llik_ocflux =  llik_ocflux + logpdf(Normal(model_ocflux[index], calibration_data[index, :obs_ocflux_err]), calibration_data[index, :obs_ocflux])
            end
        end

        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration (instrumental)
        #-----------------------------------------------------------------------
        llik_ch4inst = 0.
        if assim_ch4inst
            for (i, index)=enumerate(obs_ch4inst_indices)
                ch4inst_residual[i] = calibration_data[index, :obs_ch4inst] - model_ch4[index]
            end
            llik_ch4inst = hetero_logl_ar1(ch4inst_residual, σ_ch4inst, ρ_ch4inst, calibration_data[obs_ch4inst_indices, :ch4inst_sigma])
        end

        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration (ice core) - i.i.d blocks
        #-----------------------------------------------------------------------
        llik_ch4ice_iid = 0.
        if assim_ch4ice
            for (i,index) = enumerate(ch4ice_iid_indices)
                # Do first observation separately (not 4 previous years in calibration period for first observation).
                if i == 1
                    ch4ice_mean[i] = mean(model_ch4[index + (-1:2)])
                else
                    ch4ice_mean[i] = mean(model_ch4[index + (-4:3)])
                end
                llik_ch4ice_iid = llik_ch4ice_iid + logpdf(Normal(ch4ice_mean[i], sqrt(σ_ch4ice^2 + calibration_data[index, :ch4ice_sigma]^2)), calibration_data[index, :obs_ch4ice])
            end
        end

        #-----------------------------------------------------------------------
        # Atmospheric CH₄ Concentration (ice core) - AR1 blocks
        #-----------------------------------------------------------------------
        llik_ch4ice_ar1 = 0.
        # Use counter to track residuals for various blocks.
        counter =0
        if assim_ch4ice
            # Loop through every AR(1) block.
            for block = 1:length(ch4ice_ar1_start)
                # Within a block, calculate residuals (8 year average model result, centered on year of ice core observation).
                for index in collect(ch4ice_ar1_start[block]:ch4ice_ar1_end[block])
                    counter+=1
                    ch4ice_ar1_residual[counter] = calibration_data[index, :obs_ch4ice] - mean(model_ch4[index + (-4:3)])
                end
                # Calculate log likelihood for that block and add to total likelihood for ch4 AR(1) ice core data. (Do annoying indexing to avoid having to allocate a new residual array for every block).
                llik_ch4ice_ar1 = llik_ch4ice_ar1 + hetero_logl_ar1(ch4ice_ar1_residual[(counter-length(ch4ice_ar1_start[block]:ch4ice_ar1_end[block])+1):counter], σ_ch4ice, ρ_ch4ice, calibration_data[ch4ice_ar1_start[block]:ch4ice_ar1_end[block], :ch4ice_sigma])
            end
        end

		llik = llik_temp + llik_ocheat + llik_co2inst + llik_co2ice + llik_ocflux + llik_ch4inst + llik_ch4ice_iid + llik_ch4ice_ar1

		return llik
	end

	# (log) posterior distribution:  posterior ~ likelihood * prior
	function log_post(p)
		lpri = log_pri(p)
		lpost = isfinite(lpri) ? sneays_log_lik(p) + lpri : -Inf
		return lpost
	end

	return log_post
end
