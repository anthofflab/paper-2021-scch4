# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # Temporary file to test out climate model runs with the CAR(1) likelihood function.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

# Load required Julia packages.
using CSVFiles
using DataFrames
using Distributions
using LinearAlgebra
using Mimi
using RobustAdaptiveMetropolisSampler

# A folder with this name will be created to store all of the replication results.
results_folder_name = "my_results_CAR1"

#####################################################################################################
#####################################################################################################
# Run Social Cost of Methane Replication Code
#####################################################################################################
#####################################################################################################

# Load generic helper functions file.
include("helper_functions.jl")

# Create folder structure to save results.
build_result_folders(results_folder_name)

# Create output folder path for convenience.
output = joinpath("..", "results", results_folder_name)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Calibrate Climate Models.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Load required calibration files common to all models as well as BMA weights file.
include(joinpath("..", "calibration", "calibration_helper_functions.jl"))
include(joinpath("..", "calibration", "create_log_posterior.jl"))

# Set final year for model calibration.
calibration_end_year = 2017

# The length of the final chain (i.e. number of samples from joint posterior pdf after discarding burn-in period values).
final_chain_length = 4_500_000

# Length of burn-in period (i.e. number of initial MCMC samples to discard).
burn_in_length = 1_000_000

# Load inital conditions for all models.
initial_parameters = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "calibration_initial_values.csv"), skiplines_begin=7))

# Select initial MCMC algorithm step size (set to 5% of difference between upper and lower parameter bounds).
mcmc_step_size = (initial_parameters[:, :upper_bound] .- initial_parameters[:, :lower_bound]) * 0.05

# Select number of total samples (final samples + burn-in).
n_mcmc_samples = Int(final_chain_length + burn_in_length)

# Create equally-spaced indices to thin chains down to 10,000 and 100,000 samples.
thin_indices_100k = trunc.(Int64, collect(range(1, stop=final_chain_length, length=100_000)))
thin_indices_10k  = trunc.(Int64, collect(range(1, stop=final_chain_length, length=10_000)))

#-------------------
# SNEASY+Hector-CH4
# ------------------

println("Begin calibrating SNEASY+Hector-CH4.")

# Load model file.
include(joinpath("..", "calibration", "run_climate_models", "run_sneasy_hectorch4.jl"))

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params = sum(initial_parameters.sneasy_hector .!== missing)
initial_params_hector = convert(Array{Float64,1}, initial_parameters.sneasy_hector[1:n_params])

# Create `run_sneasy_hectorch4` function used in log-posterior calculations.
run_sneasy_hectorch4! = construct_run_sneasy_hectorch4(calibration_end_year)

# Create log-posterior function for S-hector.
log_posterior_hectorch4 = construct_log_posterior(run_sneasy_hectorch4!, :sneasy_hector, end_year=calibration_end_year)

# Carry out Bayesian calibration of S-hector using robust adaptive metropolis MCMC algorithm.
chain_hectorch4, accept_rate_hectorch4, cov_matrix_hectorch4 = RAM_sample(log_posterior_hectorch4, initial_params_hector, Diagonal(mcmc_step_size[1:n_params]), n_mcmc_samples, opt_α=0.234)

# Discard burn-in values.
burned_chain_hectorch4 = chain_hectorch4[Int(burn_in_length+1):end, :]

# Calculate mean posterior parameter values.
mean_hectorch4 = vec(mean(burned_chain_hectorch4, dims=1))

# Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
thin100k_chain_hectorch4 = DataFrame(burned_chain_hectorch4[thin_indices_100k, :])
thin10k_chain_hectorch4  = DataFrame(burned_chain_hectorch4[thin_indices_10k, :])
names!(thin100k_chain_hectorch4, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4)])
names!(thin10k_chain_hectorch4,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4)])

#-----------------------------------
# Save Calibrated Parameter Samples
# ----------------------------------
save(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mcmc_acceptance_rate.csv"), DataFrame(hector_acceptance=accept_rate_hectorch4))
save(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mean_parameters.csv"), DataFrame(parameter = initial_parameters.parameter[1:length(mean_hectorch4)], hector_mean=mean_hectorch4))
save(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_10k.csv"), DataFrame(thin10k_chain_hectorch4))
save(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_100k.csv"), DataFrame(thin100k_chain_hectorch4))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Load Data and Settings Common to Alll Climate Projection Scenarios.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Scenario settings.
pulse_year       = 2020
pulse_size       = 1.0e-6
low_ci_interval  = 0.95
high_ci_interval = 0.98

# Load calibrated parameters for each climate model.
hector_posterior_params = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_10k.csv"))))

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Calculate Climate Projections for Baseline Scenario.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

println("Begin baseline climate projections for SNEASY-Hector.")

# Set RCP scenario.
rcp_scenario = "RCP85"

# Load file to create baseline projection functions for each climate model.
include(joinpath("climate_projections", "sneasych4_baseline_case.jl"))

# Create a function for each climate model to make baseline projections.
hector_baseline_climate = construct_sneasych4_baseline_case(:sneasy_hector, rcp_scenario, pulse_year, pulse_size, 2300)

#------------------------------------
# Make baseline climate projections.
#------------------------------------

# SNEASY-Hector
hector_base_temp_baseline, hector_base_co2_baseline, hector_base_ch4_baseline, hector_base_ocean_heat_baseline,
hector_base_oceanco2_baseline, hector_pulse_temperature_baseline, hector_pulse_co2_baseline,
hector_ci_temperature_baseline, hector_ci_co2_baseline, hector_ci_ocean_heat_baseline, hector_ci_oceanco2_baseline,
hector_ci_ch4_baseline = hector_baseline_climate(hector_posterior_params, low_ci_interval, high_ci_interval)

#---------------------------------------------------
# Save baseline climate projections for each model.
#---------------------------------------------------
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_temperature.csv"), DataFrame(hector_base_temp_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_co2.csv"), DataFrame(hector_base_co2_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_ch4.csv"), DataFrame(hector_base_ch4_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_ocean_heat.csv"), DataFrame(hector_base_ocean_heat_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_oceanco2_flux.csv"), DataFrame(hector_base_oceanco2_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_temperature.csv"), DataFrame(hector_pulse_temperature_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_co2.csv"), DataFrame(hector_pulse_co2_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_temperature.csv"), DataFrame(hector_ci_temperature_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_co2.csv"), DataFrame(hector_ci_co2_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_ch4.csv"), DataFrame(hector_ci_ch4_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_ocean_heat.csv"), DataFrame(hector_ci_ocean_heat_baseline))
save(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_oceanco2_flux.csv"), DataFrame(hector_ci_oceanco2_baseline))


#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
# Calculate Climate Projections for Scenario Sampling U.S. Climate Sensitivity Distribution.
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

println("Begin climate projections for SNEASY-Hector model while sampling ECS values from the U.S. ECS distribution.")

# Set RCP scenario.
rcp_scenario = "RCP85"

# Load file to create baseline projection functions for each climate model.
include(joinpath("climate_projections", "sneasych4_us_ecs.jl"))

# Load mean posterior parameter values.
hector_posterior_means = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mean_parameters.csv"))).hector_mean

# Create a function for each climate model to make baseline projections.
hector_ecs_climate = construct_sneasych4_ecs(:sneasy_hector, rcp_scenario, pulse_year, pulse_size, 2300)

# Create a sample from the Roe & Baker climate sensitivty distribution using U.S. calibration.
norm_dist = Truncated(Normal(0.6198, 0.1841), -0.2, 0.88)
ecs_sample = 1.2 ./ (1 .- rand(norm_dist, 10_000))

#----------------------------------------------------------
# Make climate projections with Roe Baker ECS.
#----------------------------------------------------------
# SNEASY-Hector
hector_base_temp_ecs, hector_base_co2_ecs, hector_base_ch4_ecs, hector_base_ocean_heat_ecs,
hector_base_oceanco2_ecs, hector_pulse_temperature_ecs, hector_pulse_co2_ecs, hector_ci_temperature_ecs,
hector_ci_co2_ecs, hector_ci_ocean_heat_ecs, hector_ci_oceanco2_ecs, hector_ci_ch4_ecs, hector_error_indices_ecs,
hector_good_indices_ecs, hector_ecs_sample = hector_ecs_climate(ecs_sample, hector_posterior_means, low_ci_interval, high_ci_interval)

#---------------------------------------------------
# Save U.S. ECS climate projections for each model.
#---------------------------------------------------
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_temperature.csv"), DataFrame(hector_base_temp_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_co2.csv"), DataFrame(hector_base_co2_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_ch4.csv"), DataFrame(hector_base_ch4_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_ocean_heat.csv"), DataFrame(hector_base_ocean_heat_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_oceanco2_flux.csv"), DataFrame(hector_base_oceanco2_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_temperature.csv"), DataFrame(hector_pulse_temperature_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_co2.csv"), DataFrame(hector_pulse_co2_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_temperature_oldmeans_noCARcorr.csv"), DataFrame(hector_ci_temperature_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_co2_oldmeans_noCARcorr.csv"), DataFrame(hector_ci_co2_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_ch4_oldmeans_noCARcorr.csv"), DataFrame(hector_ci_ch4_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_ocean_heat_oldmeans_noCARcorr.csv"), DataFrame(hector_ci_ocean_heat_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_oceanco2_flux_oldmeans_noCARcorr.csv"), DataFrame(hector_ci_oceanco2_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "good_indices.csv"), DataFrame(indices=hector_good_indices_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "error_indices.csv"), DataFrame(indices=hector_error_indices_ecs))
save(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ecs_sample.csv"), DataFrame(ecs_samples=hector_ecs_sample))

println("All done.")
