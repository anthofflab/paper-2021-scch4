# Load required Julia packages.
using LinearAlgebra
using DataFrames
using Distributions
using CSVFiles
using RobustAdaptiveMetropolisSampler

# Load required files.
include("calibration_helper_functions.jl")
include("create_log_posterior.jl")


#------------------------------------------------------------------------------------------------------
# Calibration Parameters To Modify.
#------------------------------------------------------------------------------------------------------

# Select the climate model to use (options = :sneasy_fair, :sneasy_fund, :sneasy_hector, & :sneasy_magicc).
climate_model = :sneasy_fund

# The length of the final chain (i.e. number of samples from joint posterior pdf after discarding burn-in period values).
final_chain_length = 1_000

# Length of burn-in period (i.e. number of initial MCMC samples to discard).
burn_in_length = final_chain_length * 0.1

# Final year to carry out model calibration (by default, models initialize in 1765 and observations begin in 1850).
calibration_end_year = 2017


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
# Run Everything and Save Key Results (*no need to modify code below this line).
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# Load file with inital conditions for all models.
initial_parameters = DataFrame(load(joinpath(@__DIR__, "../", "data", "calibration_data", "calibration_initial_values.csv"), skiplines_begin=7))


# Load function to run model and intial MCMC starting point depending on SNEASY+CH4 version selected.

#---------------------------------------
if climate_model == :sneasy_fair
#---------------------------------------

	# Select starting parameter values to initialize MCMC algorithm.
	initial_mcmc_θ = collect(skipmissing(initial_parameters.sneasy_fair))

	# Select initial MCMC algorithm step size (set to 5% of difference between upper and lower parameter bounds).
	n_params = length(initial_mcmc_θ)
	mcmc_step_size = (initial_parameters[1:n_params, :upper_bound] .- initial_parameters[1:n_params, :lower_bound]) * 0.05

	# Create `run_sneasy_fairch4` function used in log-posterior calculations.
	include(joinpath("run_climate_models", "run_sneasy_fairch4.jl"))
	run_sneasych4! = construct_run_sneasy_fairch4(calibration_end_year)

#---------------------------------------
elseif climate_model == :sneasy_fund
#---------------------------------------

	# Select starting parameter values to initialize MCMC algorithm.
	initial_mcmc_θ = collect(skipmissing(initial_parameters.sneasy_fund))

	# Select initial MCMC algorithm step size (set to 5% of difference between upper and lower parameter bounds).
	n_params = length(initial_mcmc_θ)
	mcmc_step_size = (initial_parameters[1:n_params, :upper_bound] .- initial_parameters[1:n_params, :lower_bound]) * 0.05

	# Create `run_sneasy_fairch4` function used in log-posterior calculations.
	include(joinpath("run_climate_models", "run_sneasy_fundch4.jl"))
	run_sneasych4! = construct_run_sneasy_fundch4(calibration_end_year)

#---------------------------------------
elseif climate_model == :sneasy_hector
#---------------------------------------

	# Select starting parameter values to initialize MCMC algorithm.
	initial_mcmc_θ = initial_parameters.sneasy_hector

	# Select initial MCMC algorithm step size (set to 5% of difference between upper and lower parameter bounds).
	n_params = length(initial_mcmc_θ)
	mcmc_step_size = (initial_parameters[1:n_params, :upper_bound] .- initial_parameters[1:n_params, :lower_bound]) * 0.05

	# Create `run_sneasy_hectorch4` function used in log-posterior calculations.
	include(joinpath("run_climate_models", "run_sneasy_hectorch4.jl"))
	run_sneasych4! = construct_run_sneasy_hectorch4(calibration_end_year)

#---------------------------------------
elseif climate_model == :sneasy_fund
#---------------------------------------

	error("Sneasy-FUND not coded up yet.")

else
	error("Selected a climate model that does not exist. Available options are :sneasy_fair, :sneasy_fund, :sneasy_hector, and :sneasy_magicc")
end

# Create log-posterior target function to pass into MCMC sampler.
log_posterior = construct_log_posterior(run_sneasych4!, climate_model, end_year=calibration_end_year)

# Set the total chain length to determine how long to run MCMC algorithm.
total_chain_length = Int(final_chain_length + burn_in_length)

# Carry out Bayesian calibration of selected climate model using robust adaptive metropolis MCMC algorithm.
chain, accept_rate, cov_matrix = RAM_sample(log_posterior, initial_mcmc_θ, Diagonal(mcmc_step_size), total_chain_length, opt_α=0.234)

# Create a posterior parameter sample that discards the burn-in period samples.
# TODO

# Create thinned parameter samples based on user-specifications.
# TODO

# Save results.
# TODO
