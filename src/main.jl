# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file carries out all of the runs to replicate the results from Errickson et al., 2020.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

# Install required Julia packages if they are not already installed
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using CSVFiles
using DataFrames
using Distributions
using Interpolations
using LinearAlgebra
using Statistics
using Mimi
import MimiDICE2013
import MimiFAIR
import MimiFUND
import MimiHector
import MimiMAGICC
import MimiSNEASY
using RobustAdaptiveMetropolisSampler
using Distributed

# Load required Julia packages.
@everywhere begin
    using CSVFiles
    using DataFrames
    using Distributions
    using Interpolations
    using LinearAlgebra
    using Statistics
    using Mimi
    using RobustAdaptiveMetropolisSampler
    using Distributed
end

include("workflow_helper.jl")

test_run = "--testrun" in ARGS ? true : false

@info "Doing a $(test_run ? "test" : "normal") run."

central_include_filepath = joinpath(@__DIR__, "central_includes.jl")
@everywhere include($central_include_filepath)

# A folder with this name will be created to store all of the replication results.
results_folder_name = test_run ? "my_results_test" : "my_results"

#####################################################################################################
#####################################################################################################
#                           Run Social Cost of Methane Replication Code                             #
#####################################################################################################
#####################################################################################################

# Create folder structure to save results.
build_result_folders(results_folder_name)

# Create output folder path for convenience.
output = joinpath("..", "results", results_folder_name)

#######################################################################
#---------------------------------------------------------------------#
#                    Calibrate Climate Models                         #
#---------------------------------------------------------------------#
#######################################################################

println("\nBEGIN CALIBRATING MODELS.\n")

# Set final year for model calibration.
calibration_end_year = 2017

# The length of the final chain (i.e. number of samples from joint posterior pdf after discarding burn-in period values).
final_chain_length = test_run ? 10_000 : 5_000_000

# Length of burn-in period (i.e. number of initial MCMC samples to discard).
burn_in_length = test_run ? 5_000 : final_chain_length * 0.2

# Load inital conditions for all models.
initial_parameters = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "calibration_initial_values.csv"), skiplines_begin=7))

# Select initial MCMC algorithm step size (set to 5% of difference between upper and lower parameter bounds).
mcmc_step_size = (initial_parameters[:, :upper_bound] .- initial_parameters[:, :lower_bound]) * 0.05

# Select number of total samples (final samples + burn-in).
n_mcmc_samples = Int(final_chain_length + burn_in_length)

# Create equally-spaced indices to thin chains down to 10,000 and 100,000 samples.
thin_indices_100k = trunc.(Int64, collect(range(1, stop=final_chain_length, length=test_run ? 100 : 100_000)))
thin_indices_10k  = trunc.(Int64, collect(range(1, stop=final_chain_length, length=test_run ? 10 : 10_000)))

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_fair = sum(initial_parameters.sneasy_fair .!== missing)
initial_params_fair = convert(Array{Float64,1}, initial_parameters.sneasy_fair[1:n_params_fair])

# Create `run_sneasy_fairch4` function used in log-posterior calculations.
run_sneasy_fairch4! = construct_run_sneasy_fairch4(calibration_end_year)

# Create log-posterior function for S-FAIR.
log_posterior_fairch4 = construct_log_posterior(run_sneasy_fairch4!, :sneasy_fair, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_fairch4_wider = sum(initial_parameters.sneasy_fair .!== missing)
initial_params_fair_wider = convert(Array{Float64,1}, initial_parameters.sneasy_fair[1:n_params_fairch4_wider])

# Create `run_sneasy_fairch4` function used in log-posterior calculations.
run_sneasy_fairch4_wider! = construct_run_sneasy_fairch4(calibration_end_year)

# Create log-posterior function for S-FAIR.
log_posterior_fairch4_wider = construct_log_posterior_wider(run_sneasy_fairch4_wider!, :sneasy_fair, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_fundch4 = sum(initial_parameters.sneasy_fund .!== missing)
initial_params_fundch4 = convert(Array{Float64,1}, initial_parameters.sneasy_fund[1:n_params_fundch4])

# Create `run_sneasy_fundch4` function used in log-posterior calculations.
run_sneasy_fundch4! = construct_run_sneasy_fundch4(calibration_end_year)

# Create log-posterior function for S-fund.
log_posterior_fundch4 = construct_log_posterior(run_sneasy_fundch4!, :sneasy_fund, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_fundch4_wider = sum(initial_parameters.sneasy_fund .!== missing)
initial_params_fund_wider = convert(Array{Float64,1}, initial_parameters.sneasy_fund[1:n_params_fundch4_wider])

# Create `run_sneasy_fundch4` function used in log-posterior calculations.
run_sneasy_fundch4_wider! = construct_run_sneasy_fundch4(calibration_end_year)

# Create log-posterior function for S-FUND.
log_posterior_fundch4_wider = construct_log_posterior_wider(run_sneasy_fundch4_wider!, :sneasy_fund, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_hectorch4 = sum(initial_parameters.sneasy_hector .!== missing)
initial_params_hectorch4 = convert(Array{Float64,1}, initial_parameters.sneasy_hector[1:n_params_hectorch4])

# Create `run_sneasy_hectorch4` function used in log-posterior calculations.
run_sneasy_hectorch4! = construct_run_sneasy_hectorch4(calibration_end_year)

# Create log-posterior function for S-Hector.
log_posterior_hectorch4 = construct_log_posterior(run_sneasy_hectorch4!, :sneasy_hector, end_year=calibration_end_year)
        
                
# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_hectorch4_wider = sum(initial_parameters.sneasy_hector .!== missing)
initial_params_hectorch4_wider = convert(Array{Float64,1}, initial_parameters.sneasy_hector[1:n_params_hectorch4_wider])

# Create `run_sneasy_hectorch4` function used in log-posterior calculations.
run_sneasy_hectorch4_wider! = construct_run_sneasy_hectorch4(calibration_end_year)

# Create log-posterior function for S-Hector.
log_posterior_hectorch4_wider = construct_log_posterior_wider(run_sneasy_hectorch4_wider!, :sneasy_hector, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_magiccch4 = sum(initial_parameters.sneasy_magicc .!== missing)
initial_params_magiccch4 = convert(Array{Float64,1}, initial_parameters.sneasy_magicc[1:n_params_magiccch4])

# Create `run_sneasy_magiccch4` function used in log-posterior calculations.
run_sneasy_magiccch4! = construct_run_sneasy_magiccch4(calibration_end_year)

# Create log-posterior function for S-MAGICC.
log_posterior_magiccch4 = construct_log_posterior(run_sneasy_magiccch4!, :sneasy_magicc, end_year=calibration_end_year)

# Calculate number of uncertain parameters and remove "missing" values from initial parameters.
n_params_magiccch4_wider = sum(initial_parameters.sneasy_magicc .!== missing)
initial_params_magiccch4_wider = convert(Array{Float64,1}, initial_parameters.sneasy_magicc[1:n_params_magiccch4_wider])

# Create `run_sneasy_magiccch4` function used in log-posterior calculations.
run_sneasy_magiccch4_wider! = construct_run_sneasy_magiccch4(calibration_end_year)

# Create log-posterior function for S-MAGICC.
log_posterior_magiccch4_wider = construct_log_posterior_wider(run_sneasy_magiccch4_wider!, :sneasy_magicc, end_year=calibration_end_year)
   
@sync begin

    #-------------------------------#
    #-------------------------------#
    #------- SNEASY+FAIR-CH4 -------#
    #-------------------------------#
    #-------------------------------#



    @workunit "SNEASY+FAIR-CH4 - Baseline" begin
        #----------------------------------#
        #------ Baseline Calibration ------#
        #----------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_fair = sum(initial_parameters.sneasy_fair .!== missing)
        initial_params_fair = convert(Array{Float64,1}, initial_parameters.sneasy_fair[1:n_params_fair])

        # Create `run_sneasy_fairch4` function used in log-posterior calculations.
        run_sneasy_fairch4! = construct_run_sneasy_fairch4(calibration_end_year)

        # Create log-posterior function for S-FAIR.
        log_posterior_fairch4 = construct_log_posterior(run_sneasy_fairch4!, :sneasy_fair, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-FAIR using robust adaptive metropolis MCMC algorithm.
        chain_fairch4, accept_rate_fairch4, cov_matrix_fairch4 = RAM_sample(log_posterior_fairch4, initial_params_fair, Diagonal(mcmc_step_size[1:n_params_fair]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_fairch4 = chain_fairch4[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_fairch4 = vec(mean(burned_chain_fairch4, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_fairch4 = DataFrame(burned_chain_fairch4[thin_indices_100k, :])
        thin10k_chain_fairch4  = DataFrame(burned_chain_fairch4[thin_indices_10k, :])
        names!(thin100k_chain_fairch4, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fairch4)])
        names!(thin10k_chain_fairch4,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fairch4)])
    end [
        DataFrame(fair_acceptance=accept_rate_fairch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_fairch4)], fair_mean=mean_fairch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "mean_parameters.csv"),
        DataFrame(thin10k_chain_fairch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "parameters_10k.csv"),
        DataFrame(thin100k_chain_fairch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "parameters_100k.csv"),
    ]

    

    @workunit "SNEASY+FAIR-CH4 - Wider priors" begin
        #-------------------------------------#
        #------ Wider Prior Calibration ------#
        #-------------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_fairch4_wider = sum(initial_parameters.sneasy_fair .!== missing)
        initial_params_fair_wider = convert(Array{Float64,1}, initial_parameters.sneasy_fair[1:n_params_fairch4_wider])

        # Create `run_sneasy_fairch4` function used in log-posterior calculations.
        run_sneasy_fairch4_wider! = construct_run_sneasy_fairch4(calibration_end_year)

        # Create log-posterior function for S-FAIR.
        log_posterior_fairch4_wider = construct_log_posterior_wider(run_sneasy_fairch4_wider!, :sneasy_fair, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-FAIR using robust adaptive metropolis MCMC algorithm.
        chain_fairch4_wider, accept_rate_fairch4_wider, cov_matrix_fairch4_wider = RAM_sample(log_posterior_fairch4_wider, initial_params_fair_wider, Diagonal(mcmc_step_size[1:n_params_fairch4_wider]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_fairch4_wider = chain_fairch4_wider[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_fairch4_wider = vec(mean(burned_chain_fairch4_wider, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_fairch4_wider = DataFrame(burned_chain_fairch4_wider[thin_indices_100k, :])
        thin10k_chain_fairch4_wider  = DataFrame(burned_chain_fairch4_wider[thin_indices_10k, :])
        names!(thin100k_chain_fairch4_wider, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fairch4_wider)])
        names!(thin10k_chain_fairch4_wider,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fairch4_wider)])
    end [
        DataFrame(fair_acceptance=accept_rate_fairch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fair", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_fairch4_wider)], fair_mean=mean_fairch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fair", "mean_parameters.csv"),
        DataFrame(thin10k_chain_fairch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fair", "parameters_10k.csv"),
        DataFrame(thin100k_chain_fairch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fair", "parameters_100k.csv"),
    ]

    #-------------------------------#
    #-------------------------------#
    #------- SNEASY+FUND-CH4 -------#
    #-------------------------------#
    #-------------------------------#



    @workunit "SNEASY+FUND-CH4 - Baseline" begin
        #----------------------------------#
        #------ Baseline Calibration ------#
        #----------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_fundch4 = sum(initial_parameters.sneasy_fund .!== missing)
        initial_params_fundch4 = convert(Array{Float64,1}, initial_parameters.sneasy_fund[1:n_params_fundch4])

        # Create `run_sneasy_fundch4` function used in log-posterior calculations.
        run_sneasy_fundch4! = construct_run_sneasy_fundch4(calibration_end_year)

        # Create log-posterior function for S-fund.
        log_posterior_fundch4 = construct_log_posterior(run_sneasy_fundch4!, :sneasy_fund, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-fund using robust adaptive metropolis MCMC algorithm.
        chain_fundch4, accept_rate_fundch4, cov_matrix_fundch4 = RAM_sample(log_posterior_fundch4, initial_params_fundch4, Diagonal(mcmc_step_size[1:n_params_fundch4]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_fundch4 = chain_fundch4[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_fundch4 = vec(mean(burned_chain_fundch4, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_fundch4 = DataFrame(burned_chain_fundch4[thin_indices_100k, :])
        thin10k_chain_fundch4  = DataFrame(burned_chain_fundch4[thin_indices_10k, :])
        names!(thin100k_chain_fundch4, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fundch4)])
        names!(thin10k_chain_fundch4,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fundch4)])
    end [
        DataFrame(fund_acceptance=accept_rate_fundch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_fundch4)], fund_mean=mean_fundch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "mean_parameters.csv"),
        DataFrame(thin10k_chain_fundch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "parameters_10k.csv"),
        DataFrame(thin100k_chain_fundch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "parameters_100k.csv"),
    ]

    

    @workunit "SNEASY+FUND-CH4 - Wider priors" begin
        #-------------------------------------#
        #------ Wider Prior Calibration ------#
        #-------------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_fundch4_wider = sum(initial_parameters.sneasy_fund .!== missing)
        initial_params_fund_wider = convert(Array{Float64,1}, initial_parameters.sneasy_fund[1:n_params_fundch4_wider])

        # Create `run_sneasy_fundch4` function used in log-posterior calculations.
        run_sneasy_fundch4_wider! = construct_run_sneasy_fundch4(calibration_end_year)

        # Create log-posterior function for S-FUND.
        log_posterior_fundch4_wider = construct_log_posterior_wider(run_sneasy_fundch4_wider!, :sneasy_fund, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-FUND using robust adaptive metropolis MCMC algorithm.
        chain_fundch4_wider, accept_rate_fundch4_wider, cov_matrix_fundch4_wider = RAM_sample(log_posterior_fundch4_wider, initial_params_fund_wider, Diagonal(mcmc_step_size[1:n_params_fundch4_wider]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_fundch4_wider = chain_fundch4_wider[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_fundch4_wider = vec(mean(burned_chain_fundch4_wider, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_fundch4_wider = DataFrame(burned_chain_fundch4_wider[thin_indices_100k, :])
        thin10k_chain_fundch4_wider  = DataFrame(burned_chain_fundch4_wider[thin_indices_10k, :])
        names!(thin100k_chain_fundch4_wider, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fundch4_wider)])
        names!(thin10k_chain_fundch4_wider,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_fundch4_wider)])
    end [
        DataFrame(fund_acceptance=accept_rate_fundch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fund", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_fundch4_wider)], fund_mean=mean_fundch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fund", "mean_parameters.csv"),
        DataFrame(thin10k_chain_fundch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fund", "parameters_10k.csv"),
        DataFrame(thin100k_chain_fundch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fund", "parameters_100k.csv"),
    ]

    #-------------------------------#
    #-------------------------------#
    #------ SNEASY+Hector-CH4 ------#
    #-------------------------------#
    #-------------------------------#



    @workunit "SNEASY+Hector-CH4 - Baseline" begin
        #----------------------------------#
        #------ Baseline Calibration ------#
        #----------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_hectorch4 = sum(initial_parameters.sneasy_hector .!== missing)
        initial_params_hectorch4 = convert(Array{Float64,1}, initial_parameters.sneasy_hector[1:n_params_hectorch4])

        # Create `run_sneasy_hectorch4` function used in log-posterior calculations.
        run_sneasy_hectorch4! = construct_run_sneasy_hectorch4(calibration_end_year)

        # Create log-posterior function for S-Hector.
        log_posterior_hectorch4 = construct_log_posterior(run_sneasy_hectorch4!, :sneasy_hector, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-Hector using robust adaptive metropolis MCMC algorithm.
        chain_hectorch4, accept_rate_hectorch4, cov_matrix_hectorch4 = RAM_sample(log_posterior_hectorch4, initial_params_hectorch4, Diagonal(mcmc_step_size[1:n_params_hectorch4]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_hectorch4 = chain_hectorch4[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_hectorch4 = vec(mean(burned_chain_hectorch4, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_hectorch4 = DataFrame(burned_chain_hectorch4[thin_indices_100k, :])
        thin10k_chain_hectorch4  = DataFrame(burned_chain_hectorch4[thin_indices_10k, :])
        names!(thin100k_chain_hectorch4, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4)])
        names!(thin10k_chain_hectorch4,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4)])
    end [
        DataFrame(hector_acceptance=accept_rate_hectorch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_hectorch4)], hector_mean=mean_hectorch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mean_parameters.csv"),
        DataFrame(thin10k_chain_hectorch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_10k.csv"),
        DataFrame(thin100k_chain_hectorch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_100k.csv"),
    ]

    

    @workunit "SNEASY+Hector-CH4 - Wider priors" begin
        #-------------------------------------#
        #------ Wider Prior Calibration ------#
        #-------------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_hectorch4_wider = sum(initial_parameters.sneasy_hector .!== missing)
        initial_params_hectorch4_wider = convert(Array{Float64,1}, initial_parameters.sneasy_hector[1:n_params_hectorch4_wider])

        # Create `run_sneasy_hectorch4` function used in log-posterior calculations.
        run_sneasy_hectorch4_wider! = construct_run_sneasy_hectorch4(calibration_end_year)

        # Create log-posterior function for S-Hector.
        log_posterior_hectorch4_wider = construct_log_posterior_wider(run_sneasy_hectorch4_wider!, :sneasy_hector, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-Hector using robust adaptive metropolis MCMC algorithm.
        chain_hectorch4_wider, accept_rate_hectorch4_wider, cov_matrix_hectorch4_wider = RAM_sample(log_posterior_hectorch4_wider, initial_params_hectorch4_wider, Diagonal(mcmc_step_size[1:n_params_hectorch4_wider]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_hectorch4_wider = chain_hectorch4_wider[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_hectorch4_wider = vec(mean(burned_chain_hectorch4_wider, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_hectorch4_wider = DataFrame(burned_chain_hectorch4_wider[thin_indices_100k, :])
        thin10k_chain_hectorch4_wider  = DataFrame(burned_chain_hectorch4_wider[thin_indices_10k, :])
        names!(thin100k_chain_hectorch4_wider, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4_wider)])
        names!(thin10k_chain_hectorch4_wider,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_hectorch4_wider)])
    end [
        DataFrame(hector_acceptance=accept_rate_hectorch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_hector", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_hectorch4_wider)], hector_mean=mean_hectorch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_hector", "mean_parameters.csv"),
        DataFrame(thin10k_chain_hectorch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_hector", "parameters_10k.csv"),
        DataFrame(thin100k_chain_hectorch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_hector", "parameters_100k.csv"),
    ]

    #-------------------------------#
    #-------------------------------#
    #------ SNEASY+MAGICC-CH4 ------#
    #-------------------------------#
    #-------------------------------#



    @workunit "SNEASY+MAGICC-CH4 - Baseline" begin
        #----------------------------------#
        #------ Baseline Calibration ------#
        #----------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_magiccch4 = sum(initial_parameters.sneasy_magicc .!== missing)
        initial_params_magiccch4 = convert(Array{Float64,1}, initial_parameters.sneasy_magicc[1:n_params_magiccch4])

        # Create `run_sneasy_magiccch4` function used in log-posterior calculations.
        run_sneasy_magiccch4! = construct_run_sneasy_magiccch4(calibration_end_year)

        # Create log-posterior function for S-MAGICC.
        log_posterior_magiccch4 = construct_log_posterior(run_sneasy_magiccch4!, :sneasy_magicc, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-MAGICC using robust adaptive metropolis MCMC algorithm.
        chain_magiccch4, accept_rate_magiccch4, cov_matrix_magiccch4 = RAM_sample(log_posterior_magiccch4, initial_params_magiccch4, Diagonal(mcmc_step_size[1:n_params_magiccch4]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_magiccch4 = chain_magiccch4[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_magiccch4 = vec(mean(burned_chain_magiccch4, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_magiccch4 = DataFrame(burned_chain_magiccch4[thin_indices_100k, :])
        thin10k_chain_magiccch4  = DataFrame(burned_chain_magiccch4[thin_indices_10k, :])
        names!(thin100k_chain_magiccch4, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_magiccch4)])
        names!(thin10k_chain_magiccch4,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_magiccch4)])
    end [
        DataFrame(magicc_acceptance=accept_rate_magiccch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_magiccch4)], magicc_mean=mean_magiccch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "mean_parameters.csv"),
        DataFrame(thin10k_chain_magiccch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "parameters_10k.csv"),
        DataFrame(thin100k_chain_magiccch4) => joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "parameters_100k.csv"),
    ]

    

    @workunit "SNEASY+MAGICC-CH4 - Wider priors" begin
        #-------------------------------------#
        #------ Wider Prior Calibration ------#
        #-------------------------------------#

        # Calculate number of uncertain parameters and remove "missing" values from initial parameters.
        n_params_magiccch4_wider = sum(initial_parameters.sneasy_magicc .!== missing)
        initial_params_magiccch4_wider = convert(Array{Float64,1}, initial_parameters.sneasy_magicc[1:n_params_magiccch4_wider])

        # Create `run_sneasy_magiccch4` function used in log-posterior calculations.
        run_sneasy_magiccch4_wider! = construct_run_sneasy_magiccch4(calibration_end_year)

        # Create log-posterior function for S-MAGICC.
        log_posterior_magiccch4_wider = construct_log_posterior_wider(run_sneasy_magiccch4_wider!, :sneasy_magicc, end_year=calibration_end_year)

        # Carry out Bayesian calibration of S-HMAGICC using robust adaptive metropolis MCMC algorithm.
        chain_magiccch4_wider, accept_rate_magiccch4_wider, cov_matrix_magiccch4_wider = RAM_sample(log_posterior_magiccch4_wider, initial_params_magiccch4_wider, Diagonal(mcmc_step_size[1:n_params_magiccch4_wider]), n_mcmc_samples, opt_α=0.234)

        # Discard burn-in values.
        burned_chain_magiccch4_wider = chain_magiccch4_wider[Int(burn_in_length+1):end, :]

        # Calculate mean posterior parameter values.
        mean_magiccch4_wider = vec(mean(burned_chain_magiccch4_wider, dims=1))

        # Create thinned chains (after burn-in period) with 10,000 and 100,000 samples and assign parameter names to each column.
        thin100k_chain_magiccch4_wider = DataFrame(burned_chain_magiccch4_wider[thin_indices_100k, :])
        thin10k_chain_magiccch4_wider  = DataFrame(burned_chain_magiccch4_wider[thin_indices_10k, :])
        names!(thin100k_chain_magiccch4_wider, [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_magiccch4_wider)])
        names!(thin10k_chain_magiccch4_wider,  [Symbol(initial_parameters.parameter[i]) for i in 1:length(mean_magiccch4_wider)])
    end [
        DataFrame(magicc_acceptance=accept_rate_magiccch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_magicc", "mcmc_acceptance_rate.csv"),
        DataFrame(parameter = initial_parameters.parameter[1:length(mean_magiccch4_wider)], magicc_mean=mean_magiccch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_magicc", "mean_parameters.csv"),
        DataFrame(thin10k_chain_magiccch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_magicc", "parameters_10k.csv"),
        DataFrame(thin100k_chain_magiccch4_wider) => joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_magicc", "parameters_100k.csv"),
    ]

end

@info "Done with calibration."

#---------------------------------------------#
#---------------------------------------------#
#------ Calculate and Save BMA Weights. ------#
# --------------------------------------------#
#---------------------------------------------#
if !test_run
    if !isfile(joinpath(@__DIR__, output, "calibrated_parameters", "bma_weights", "bma_weights.csv"))
        println("STARTING: BMA")

        thin100k_chain_fairch4 = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "parameters_100k.csv")))
        thin100k_chain_fundch4 = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "parameters_100k.csv")))
        thin100k_chain_hectorch4 = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_100k.csv")))
        thin100k_chain_magiccch4 = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "parameters_100k.csv")))

        bma_weights = calculate_bma_weights(Matrix(thin100k_chain_fairch4), Matrix(thin100k_chain_fundch4), Matrix(thin100k_chain_hectorch4), Matrix(thin100k_chain_magiccch4), log_posterior_fairch4, log_posterior_fundch4, log_posterior_hectorch4, log_posterior_magiccch4)    

        println("SAVING: BMA")

        save(joinpath(@__DIR__, output, "calibrated_parameters", "bma_weights", "bma_weights.csv"), DataFrame(col1=bma_weights))

        thin100k_chain_fairch4 = nothing
        thin100k_chain_fundch4 = nothing
        thin100k_chain_hectorch4 = nothing
        thin100k_chain_magiccch4 = nothing

        println("FINISHED: BMA")
    else
        println("SKPPING: BMA")
    end
else
    println("\nNot Calculating BMA Weights for test run.\n")
end



#######################################################################
#---------------------------------------------------------------------#
#                  Climate Hindcasts & Projections                    #
#---------------------------------------------------------------------#
#######################################################################

println("\nBEGIN MAKING CLIMATE PROJECTIONS.\n")

#----------------------------------------------------------------------
# Load Data and Settings Common to All Climate Projection Scenarios.
#----------------------------------------------------------------------

# Scenario settings.
pulse_year       = 2020
pulse_size       = 1.0e-6
low_ci_interval  = 0.95
high_ci_interval = 0.98

# Load baseline calibrated parameters for each climate model.
fair_posterior_params   = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "parameters_100k.csv"))))
fund_posterior_params   = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "parameters_100k.csv"))))
hector_posterior_params = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "parameters_100k.csv"))))
magicc_posterior_params = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "parameters_100k.csv"))))

# Load calibrated parameters for each climate model using wider prior parameter distributions.
fair_posterior_params_wider   = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fair", "parameters_100k.csv"))))
fund_posterior_params_wider   = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_fund", "parameters_100k.csv"))))
hector_posterior_params_wider = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_hector", "parameters_100k.csv"))))
magicc_posterior_params_wider = convert(Array{Float64,2, }, DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "wider_priors", "s_magicc", "parameters_100k.csv"))))

# Load mean posterior parameter values.
fair_posterior_means   = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fair", "mean_parameters.csv"))).fair_mean
fund_posterior_means   = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_fund", "mean_parameters.csv"))).fund_mean
hector_posterior_means = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_hector", "mean_parameters.csv"))).hector_mean
magicc_posterior_means = DataFrame(load(joinpath(@__DIR__, output, "calibrated_parameters", "s_magicc", "mean_parameters.csv"))).magicc_mean

# Create an ECS sample from the Roe & Baker equilibrium climate sensitivty distribution using the U.S. calibration settings.
norm_dist = Truncated(Normal(0.6198, 0.1841), -0.2, 0.88)
ecs_sample = 1.2 ./ (1 .- rand(norm_dist, test_run ? 100 : 100_000))

@sync begin

    #----------------------------------------------------------------------------#
    #----------------------------------------------------------------------------#
    #------ Calculate Climate Projections for Baseline Scenario (RCP 8.5). ------#
    #----------------------------------------------------------------------------#
    #----------------------------------------------------------------------------#

    @workunit "Baseline climate projections for SNEASY+FAIR-CH4" begin
        fair_baseline_climate   = construct_sneasych4_baseline_case(:sneasy_fair, "RCP85", pulse_year, pulse_size, 2300)

        fair_base_temperature_baseline, fair_base_co2_baseline, fair_base_ch4_baseline, fair_base_ocean_heat_baseline,
        fair_base_oceanco2_baseline, fair_pulse_temperature_baseline, fair_pulse_co2_baseline, fair_ci_temperature_baseline,
        fair_ci_co2_baseline, fair_ci_ocean_heat_baseline, fair_ci_oceanco2_baseline, fair_ci_ch4_baseline, fair_error_indices_baseline,
        fair_good_indices_baseline = fair_baseline_climate(fair_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_error_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "error_indices.csv"),
        DataFrame(indices=fair_good_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "good_indices.csv"),

    ]

    @workunit "Baseline climate projections for SNEASY+FUND-CH4." begin
        fund_baseline_climate   = construct_sneasych4_baseline_case(:sneasy_fund, "RCP85", pulse_year, pulse_size, 2300)

        fund_base_temperature_baseline, fund_base_co2_baseline, fund_base_ch4_baseline, fund_base_ocean_heat_baseline,
        fund_base_oceanco2_baseline, fund_pulse_temperature_baseline, fund_pulse_co2_baseline, fund_ci_temperature_baseline,
        fund_ci_co2_baseline, fund_ci_ocean_heat_baseline, fund_ci_oceanco2_baseline, fund_ci_ch4_baseline, fund_error_indices_baseline,
        fund_good_indices_baseline = fund_baseline_climate(fund_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_error_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "good_indices.csv"),
    ]

    @workunit "Baseline climate projections for SNEASY+Hector-CH4." begin
        hector_baseline_climate = construct_sneasych4_baseline_case(:sneasy_hector, "RCP85", pulse_year, pulse_size, 2300)

        hector_base_temperature_baseline, hector_base_co2_baseline, hector_base_ch4_baseline, hector_base_ocean_heat_baseline,
        hector_base_oceanco2_baseline, hector_pulse_temperature_baseline, hector_pulse_co2_baseline, hector_ci_temperature_baseline,
        hector_ci_co2_baseline, hector_ci_ocean_heat_baseline, hector_ci_oceanco2_baseline, hector_ci_ch4_baseline, hector_error_indices_baseline,
        hector_good_indices_baseline = hector_baseline_climate(hector_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_temperature.csv"),
        DataFrame(hector_ci_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_error_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "error_indices.csv"),
        DataFrame(indices=hector_good_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "good_indices.csv"),
    ]

    @workunit "Baseline climate projections for SNEASY+MAGICC-CH4." begin
        magicc_baseline_climate = construct_sneasych4_baseline_case(:sneasy_magicc, "RCP85", pulse_year, pulse_size, 2300)

        magicc_base_temperature_baseline, magicc_base_co2_baseline, magicc_base_ch4_baseline, magicc_base_ocean_heat_baseline,
        magicc_base_oceanco2_baseline, magicc_pulse_temperature_baseline, magicc_pulse_co2_baseline, magicc_ci_temperature_baseline,
        magicc_ci_co2_baseline, magicc_ci_ocean_heat_baseline, magicc_ci_oceanco2_baseline, magicc_ci_ch4_baseline, magicc_error_indices_baseline,
        magicc_good_indices_baseline = magicc_baseline_climate(magicc_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_error_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "error_indices.csv"),
        DataFrame(indices=magicc_good_indices_baseline) => joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "good_indices.csv"),
    ]

    @workunit "RCP2.6 climate projections for SNEASY+FAIR-CH4" begin
        fair_rcp26_climate   = construct_sneasych4_baseline_case(:sneasy_fair, "RCP26", pulse_year, pulse_size, 2300)

        fair_base_temperature_rcp26, fair_base_co2_rcp26, fair_base_ch4_rcp26, fair_base_ocean_heat_rcp26,
        fair_base_oceanco2_rcp26, fair_pulse_temperature_rcp26, fair_pulse_co2_rcp26, fair_ci_temperature_rcp26,
        fair_ci_co2_rcp26, fair_ci_ocean_heat_rcp26, fair_ci_oceanco2_rcp26, fair_ci_ch4_rcp26, fair_error_indices_rcp26,
        fair_good_indices_rcp26 = fair_rcp26_climate(fair_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_error_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "error_indices.csv"),
        DataFrame(indices=fair_good_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "good_indices.csv"),
    ]

    @workunit "RCP2.6 climate projections for SNEASY+FUND-CH4" begin
        fund_rcp26_climate   = construct_sneasych4_baseline_case(:sneasy_fund, "RCP26", pulse_year, pulse_size, 2300)

        fund_base_temperature_rcp26, fund_base_co2_rcp26, fund_base_ch4_rcp26, fund_base_ocean_heat_rcp26,
        fund_base_oceanco2_rcp26, fund_pulse_temperature_rcp26, fund_pulse_co2_rcp26, fund_ci_temperature_rcp26,
        fund_ci_co2_rcp26, fund_ci_ocean_heat_rcp26, fund_ci_oceanco2_rcp26, fund_ci_ch4_rcp26, fund_error_indices_rcp26,
        fund_good_indices_rcp26 = fund_rcp26_climate(fund_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_error_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "good_indices.csv"),
    ]

    @workunit "Begin RCP2.6 climate projections for SNEASY+Hector-CH4." begin
        hector_rcp26_climate = construct_sneasych4_baseline_case(:sneasy_hector, "RCP26", pulse_year, pulse_size, 2300)

        hector_base_temperature_rcp26, hector_base_co2_rcp26, hector_base_ch4_rcp26, hector_base_ocean_heat_rcp26,
        hector_base_oceanco2_rcp26, hector_pulse_temperature_rcp26, hector_pulse_co2_rcp26, hector_ci_temperature_rcp26,
        hector_ci_co2_rcp26, hector_ci_ocean_heat_rcp26, hector_ci_oceanco2_rcp26, hector_ci_ch4_rcp26, hector_error_indices_rcp26,
        hector_good_indices_rcp26 = hector_rcp26_climate(hector_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "ci_temperature.csv"),
        DataFrame(hector_ci_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_error_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "error_indices.csv"),
        DataFrame(indices=hector_good_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "good_indices.csv"),
    ]

    @workunit "RCP2.6 climate projections for SNEASY+MAGICC-CH4" begin
        magicc_rcp26_climate = construct_sneasych4_baseline_case(:sneasy_magicc, "RCP26", pulse_year, pulse_size, 2300)

        magicc_base_temperature_rcp26, magicc_base_co2_rcp26, magicc_base_ch4_rcp26, magicc_base_ocean_heat_rcp26,
        magicc_base_oceanco2_rcp26, magicc_pulse_temperature_rcp26, magicc_pulse_co2_rcp26, magicc_ci_temperature_rcp26,
        magicc_ci_co2_rcp26, magicc_ci_ocean_heat_rcp26, magicc_ci_oceanco2_rcp26, magicc_ci_ch4_rcp26, magicc_error_indices_rcp26,
        magicc_good_indices_rcp26 = magicc_rcp26_climate(magicc_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_error_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "error_indices.csv"),
        DataFrame(indices=magicc_good_indices_rcp26) => joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FAIR-CH4 using outdated CH₄ forcing" begin
        fair_oldrf_climate   = construct_sneasych4_outdated_forcing(:sneasy_fair, "RCP85", pulse_year, pulse_size, 2300)

        fair_base_temperature_oldrf, fair_base_co2_oldrf, fair_base_ch4_oldrf, fair_base_ocean_heat_oldrf,
        fair_base_oceanco2_oldrf, fair_pulse_temperature_oldrf, fair_pulse_co2_oldrf, fair_ci_temperature_oldrf,
        fair_ci_co2_oldrf, fair_ci_ocean_heat_oldrf, fair_ci_oceanco2_oldrf, fair_ci_ch4_oldrf, fair_error_indices_oldrf,
        fair_good_indices_oldrf = fair_oldrf_climate(fair_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_error_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "error_indices.csv"),
        DataFrame(indices=fair_good_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FUND-CH4 using outdated CH₄ forcing" begin
        fund_oldrf_climate   = construct_sneasych4_outdated_forcing(:sneasy_fund, "RCP85", pulse_year, pulse_size, 2300)

        fund_base_temperature_oldrf, fund_base_co2_oldrf, fund_base_ch4_oldrf, fund_base_ocean_heat_oldrf,
        fund_base_oceanco2_oldrf, fund_pulse_temperature_oldrf, fund_pulse_co2_oldrf, fund_ci_temperature_oldrf,
        fund_ci_co2_oldrf, fund_ci_ocean_heat_oldrf, fund_ci_oceanco2_oldrf, fund_ci_ch4_oldrf, fund_error_indices_oldrf,
        fund_good_indices_oldrf = fund_oldrf_climate(fund_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_error_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+Hector-CH4 using outdated CH₄ forcing" begin
        hector_oldrf_climate = construct_sneasych4_outdated_forcing(:sneasy_hector, "RCP85", pulse_year, pulse_size, 2300)

        hector_base_temperature_oldrf, hector_base_co2_oldrf, hector_base_ch4_oldrf, hector_base_ocean_heat_oldrf,
        hector_base_oceanco2_oldrf, hector_pulse_temperature_oldrf, hector_pulse_co2_oldrf, hector_ci_temperature_oldrf,
        hector_ci_co2_oldrf, hector_ci_ocean_heat_oldrf, hector_ci_oceanco2_oldrf, hector_ci_ch4_oldrf, hector_error_indices_oldrf,
        hector_good_indices_oldrf = hector_oldrf_climate(hector_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "ci_temperature.csv"),
        DataFrame(hector_ci_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_error_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "error_indices.csv"),
        DataFrame(indices=hector_good_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+MAGICC-CH4 using outdated CH₄ forcing" begin
        magicc_oldrf_climate = construct_sneasych4_outdated_forcing(:sneasy_magicc, "RCP85", pulse_year, pulse_size, 2300)

        magicc_base_temperature_oldrf, magicc_base_co2_oldrf, magicc_base_ch4_oldrf, magicc_base_ocean_heat_oldrf,
        magicc_base_oceanco2_oldrf, magicc_pulse_temperature_oldrf, magicc_pulse_co2_oldrf, magicc_ci_temperature_oldrf,
        magicc_ci_co2_oldrf, magicc_ci_ocean_heat_oldrf, magicc_ci_oceanco2_oldrf, magicc_ci_ch4_oldrf, magicc_error_indices_oldrf,
        magicc_good_indices_oldrf = magicc_oldrf_climate(magicc_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_error_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "error_indices.csv"),
        DataFrame(indices=magicc_good_indices_oldrf) => joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FAIR-CH4 without posterior parameter correlations" begin
        fair_remove_correlations_climate   = construct_sneasych4_remove_correlations(:sneasy_fair, "RCP85", pulse_year, pulse_size, 2300)

        fair_base_temperature_corr, fair_base_co2_corr, fair_base_ch4_corr, fair_base_ocean_heat_corr,
        fair_base_oceanco2_corr, fair_pulse_temperature_corr, fair_pulse_co2_corr, fair_ci_temperature_corr,
        fair_ci_co2_corr, fair_ci_ocean_heat_corr, fair_ci_oceanco2_corr, fair_ci_ch4_corr, fair_error_indices_corr,
        fair_good_indices_corr, fair_random_indices_corr = fair_remove_correlations_climate(fair_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_good_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "good_indices.csv"),
        DataFrame(indices=fair_error_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "error_indices.csv"),
        DataFrame(fair_random_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "random_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FUND-CH4 without posterior parameter correlations" begin
        fund_remove_correlations_climate   = construct_sneasych4_remove_correlations(:sneasy_fund, "RCP85", pulse_year, pulse_size, 2300)

        fund_base_temperature_corr, fund_base_co2_corr, fund_base_ch4_corr, fund_base_ocean_heat_corr,
        fund_base_oceanco2_corr, fund_pulse_temperature_corr, fund_pulse_co2_corr, fund_ci_temperature_corr,
        fund_ci_co2_corr, fund_ci_ocean_heat_corr, fund_ci_oceanco2_corr, fund_ci_ch4_corr, fund_error_indices_corr,
        fund_good_indices_corr, fund_random_indices_corr = fund_remove_correlations_climate(fund_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_good_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "good_indices.csv"),
        DataFrame(indices=fund_error_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "error_indices.csv"),
        DataFrame(fund_random_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "random_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+Hector-CH4 without posterior parameter correlations" begin
        hector_remove_correlations_climate = construct_sneasych4_remove_correlations(:sneasy_hector, "RCP85", pulse_year, pulse_size, 2300)

        hector_base_temperature_corr, hector_base_co2_corr, hector_base_ch4_corr, hector_base_ocean_heat_corr,
        hector_base_oceanco2_corr, hector_pulse_temperature_corr, hector_pulse_co2_corr, hector_ci_temperature_corr,
        hector_ci_co2_corr, hector_ci_ocean_heat_corr, hector_ci_oceanco2_corr, hector_ci_ch4_corr, hector_error_indices_corr,
        hector_good_indices_corr, hector_random_indices_corr = hector_remove_correlations_climate(hector_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "ci_temperature.csv"),
        DataFrame(hector_ci_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_good_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "good_indices.csv"),
        DataFrame(indices=hector_error_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "error_indices.csv"),
        DataFrame(hector_random_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "random_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+MAGICC-CH4 without posterior parameter correlations" begin
        magicc_remove_correlations_climate = construct_sneasych4_remove_correlations(:sneasy_magicc, "RCP85", pulse_year, pulse_size, 2300)

        magicc_base_temperature_corr, magicc_base_co2_corr, magicc_base_ch4_corr, magicc_base_ocean_heat_corr,
        magicc_base_oceanco2_corr, magicc_pulse_temperature_corr, magicc_pulse_co2_corr, magicc_ci_temperature_corr,
        magicc_ci_co2_corr, magicc_ci_ocean_heat_corr, magicc_ci_oceanco2_corr, magicc_ci_ch4_corr, magicc_error_indices_corr,
        magicc_good_indices_corr, magicc_random_indices_corr = magicc_remove_correlations_climate(magicc_posterior_params, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_good_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "good_indices.csv"),
        DataFrame(indices=magicc_error_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "error_indices.csv"),
        DataFrame(magicc_random_indices_corr) => joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "random_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FAIR-CH4 while sampling the U.S. climate sensitivity distribution" begin
        fair_ecs_climate   = construct_sneasych4_ecs(:sneasy_fair, "RCP85", pulse_year, pulse_size, 2300)

        fair_base_temp_ecs, fair_base_co2_ecs, fair_base_ch4_ecs, fair_base_ocean_heat_ecs,
        fair_base_oceanco2_ecs, fair_pulse_temperature_ecs, fair_pulse_co2_ecs, fair_ci_temperature_ecs,
        fair_ci_co2_ecs, fair_ci_ocean_heat_ecs, fair_ci_oceanco2_ecs, fair_ci_ch4_ecs, fair_error_indices_ecs,
        fair_good_indices_ecs, fair_ecs_sample = fair_ecs_climate(ecs_sample, fair_posterior_means, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temp_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_good_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "good_indices.csv"),
        DataFrame(indices=fair_error_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "error_indices.csv"),
        DataFrame(ecs_samples=fair_ecs_sample) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "ecs_sample.csv"),
    ]

    @workunit "Climate projections for SNEASY+FUND-CH4 while sampling the U.S. climate sensitivity distribution" begin
        fund_ecs_climate   = construct_sneasych4_ecs(:sneasy_fund, "RCP85", pulse_year, pulse_size, 2300)

        fund_base_temp_ecs, fund_base_co2_ecs, fund_base_ch4_ecs, fund_base_ocean_heat_ecs,
        fund_base_oceanco2_ecs, fund_pulse_temperature_ecs, fund_pulse_co2_ecs, fund_ci_temperature_ecs,
        fund_ci_co2_ecs, fund_ci_ocean_heat_ecs, fund_ci_oceanco2_ecs, fund_ci_ch4_ecs, fund_error_indices_ecs,
        fund_good_indices_ecs, fund_ecs_sample = fund_ecs_climate(ecs_sample, fund_posterior_means, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temp_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_good_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "good_indices.csv"),
        DataFrame(indices=fund_error_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "error_indices.csv"),
        DataFrame(ecs_samples=fund_ecs_sample) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "ecs_sample.csv"),
    ]

    @workunit "Climate projections for SNEASY+Hector-CH4 while sampling the U.S. climate sensitivity distribution" begin
        hector_ecs_climate = construct_sneasych4_ecs(:sneasy_hector, "RCP85", pulse_year, pulse_size, 2300)

        hector_base_temp_ecs, hector_base_co2_ecs, hector_base_ch4_ecs, hector_base_ocean_heat_ecs,
        hector_base_oceanco2_ecs, hector_pulse_temperature_ecs, hector_pulse_co2_ecs, hector_ci_temperature_ecs,
        hector_ci_co2_ecs, hector_ci_ocean_heat_ecs, hector_ci_oceanco2_ecs, hector_ci_ch4_ecs, hector_error_indices_ecs,
        hector_good_indices_ecs, hector_ecs_sample = hector_ecs_climate(ecs_sample, hector_posterior_means, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temp_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_temperature.csv"),
        DataFrame(hector_ci_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_good_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "good_indices.csv"),
        DataFrame(indices=hector_error_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "error_indices.csv"),
        DataFrame(ecs_samples=hector_ecs_sample) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "ecs_sample.csv"),
    ]

    @workunit "Climate projections for SNEASY+MAGICC-CH4 while sampling the U.S. climate sensitivity distribution" begin
        magicc_ecs_climate = construct_sneasych4_ecs(:sneasy_magicc, "RCP85", pulse_year, pulse_size, 2300)

        magicc_base_temp_ecs, magicc_base_co2_ecs, magicc_base_ch4_ecs, magicc_base_ocean_heat_ecs,
        magicc_base_oceanco2_ecs, magicc_pulse_temperature_ecs, magicc_pulse_co2_ecs, magicc_ci_temperature_ecs,
        magicc_ci_co2_ecs, magicc_ci_ocean_heat_ecs, magicc_ci_oceanco2_ecs, magicc_ci_ch4_ecs, magicc_error_indices_ecs,
        magicc_good_indices_ecs, magicc_ecs_sample = magicc_ecs_climate(ecs_sample, magicc_posterior_means, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temp_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_good_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "good_indices.csv"),
        DataFrame(indices=magicc_error_indices_ecs) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "error_indices.csv"),
        DataFrame(ecs_samples=magicc_ecs_sample) => joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "ecs_sample.csv"),
    ]

    @workunit "Climate projections for SNEASY+FAIR-CH4 using wider prior parameter distributions" begin
        fair_wider_climate   = construct_sneasych4_baseline_case(:sneasy_fair, "RCP85", pulse_year, pulse_size, 2300)

        fair_base_temperature_wider, fair_base_co2_wider, fair_base_ch4_wider, fair_base_ocean_heat_wider,
        fair_base_oceanco2_wider, fair_pulse_temperature_wider, fair_pulse_co2_wider, fair_ci_temperature_wider,
        fair_ci_co2_wider, fair_ci_ocean_heat_wider, fair_ci_oceanco2_wider, fair_ci_ch4_wider, fair_error_indices_wider,
        fair_good_indices_wider = fair_wider_climate(fair_posterior_params_wider, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fair_base_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_temperature.csv"),
        DataFrame(fair_base_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_co2.csv"),
        DataFrame(fair_base_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_ch4.csv"),
        DataFrame(fair_base_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_ocean_heat.csv"),
        DataFrame(fair_base_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_oceanco2_flux.csv"),
        DataFrame(fair_pulse_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_temperature.csv"),
        DataFrame(fair_pulse_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_co2.csv"),
        DataFrame(fair_ci_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "ci_temperature.csv"),
        DataFrame(fair_ci_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "ci_co2.csv"),
        DataFrame(fair_ci_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "ci_ch4.csv"),
        DataFrame(fair_ci_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "ci_ocean_heat.csv"),
        DataFrame(fair_ci_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fair_error_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "error_indices.csv"),
        DataFrame(indices=fair_good_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+FUND-CH4 using wider prior parameter distributions" begin
        fund_wider_climate   = construct_sneasych4_baseline_case(:sneasy_fund, "RCP85", pulse_year, pulse_size, 2300)

        fund_base_temperature_wider, fund_base_co2_wider, fund_base_ch4_wider, fund_base_ocean_heat_wider,
        fund_base_oceanco2_wider, fund_pulse_temperature_wider, fund_pulse_co2_wider, fund_ci_temperature_wider,
        fund_ci_co2_wider, fund_ci_ocean_heat_wider, fund_ci_oceanco2_wider, fund_ci_ch4_wider, fund_error_indices_wider,
        fund_good_indices_wider = fund_wider_climate(fund_posterior_params_wider, low_ci_interval, high_ci_interval)
    end [
        DataFrame(fund_base_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_temperature.csv"),
        DataFrame(fund_base_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_co2.csv"),
        DataFrame(fund_base_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_ch4.csv"),
        DataFrame(fund_base_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_ocean_heat.csv"),
        DataFrame(fund_base_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_oceanco2_flux.csv"),
        DataFrame(fund_pulse_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_temperature.csv"),
        DataFrame(fund_pulse_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_co2.csv"),
        DataFrame(fund_ci_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "ci_temperature.csv"),
        DataFrame(fund_ci_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "ci_co2.csv"),
        DataFrame(fund_ci_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "ci_ch4.csv"),
        DataFrame(fund_ci_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "ci_ocean_heat.csv"),
        DataFrame(fund_ci_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "ci_oceanco2_flux.csv"),
        DataFrame(indices=fund_error_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "good_indices.csv"),
    ]

    @workunit "Climate projections for SNEASY+Hector-CH4 using wider prior parameter distributions" begin
        hector_wider_climate = construct_sneasych4_baseline_case(:sneasy_hector, "RCP85", pulse_year, pulse_size, 2300)

        hector_base_temperature_wider, hector_base_co2_wider, hector_base_ch4_wider, hector_base_ocean_heat_wider,
        hector_base_oceanco2_wider, hector_pulse_temperature_wider, hector_pulse_co2_wider, hector_ci_temperature_wider,
        hector_ci_co2_wider, hector_ci_ocean_heat_wider, hector_ci_oceanco2_wider, hector_ci_ch4_wider, hector_error_indices_wider,
        hector_good_indices_wider = hector_wider_climate(hector_posterior_params_wider, low_ci_interval, high_ci_interval)
    end [
        DataFrame(hector_base_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_temperature.csv"),
        DataFrame(hector_base_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_co2.csv"),
        DataFrame(hector_base_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_ch4.csv"),
        DataFrame(hector_base_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_ocean_heat.csv"),
        DataFrame(hector_base_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_oceanco2_flux.csv"),
        DataFrame(hector_pulse_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_temperature.csv"),
        DataFrame(hector_pulse_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_co2.csv"),
        DataFrame(hector_ci_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "ci_temperature.csv"),
     DataFrame(hector_ci_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "ci_co2.csv"),
        DataFrame(hector_ci_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "ci_ch4.csv"),
        DataFrame(hector_ci_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "ci_ocean_heat.csv"),
        DataFrame(hector_ci_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "ci_oceanco2_flux.csv"),
        DataFrame(indices=hector_error_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "error_indices.csv"),
        DataFrame(indices=hector_good_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "good_indices.csv"),
    ]

    @workunit "climate projections for SNEASY+MAGICC-CH4 using wider prior parameter distributions" begin
        magicc_wider_climate = construct_sneasych4_baseline_case(:sneasy_magicc, "RCP85", pulse_year, pulse_size, 2300)

        magicc_base_temperature_wider, magicc_base_co2_wider, magicc_base_ch4_wider, magicc_base_ocean_heat_wider,
        magicc_base_oceanco2_wider, magicc_pulse_temperature_wider, magicc_pulse_co2_wider, magicc_ci_temperature_wider,
        magicc_ci_co2_wider, magicc_ci_ocean_heat_wider, magicc_ci_oceanco2_wider, magicc_ci_ch4_wider, magicc_error_indices_wider,
        magicc_good_indices_wider = magicc_wider_climate(magicc_posterior_params_wider, low_ci_interval, high_ci_interval)
    end [
        DataFrame(magicc_base_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_temperature.csv"),
        DataFrame(magicc_base_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_co2.csv"),
        DataFrame(magicc_base_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_ch4.csv"),
        DataFrame(magicc_base_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_ocean_heat.csv"),
        DataFrame(magicc_base_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_oceanco2_flux.csv"),
        DataFrame(magicc_pulse_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_temperature.csv"),
        DataFrame(magicc_pulse_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_co2.csv"),
        DataFrame(magicc_ci_temperature_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "ci_temperature.csv"),
        DataFrame(magicc_ci_co2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "ci_co2.csv"),
        DataFrame(magicc_ci_ch4_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "ci_ch4.csv"),
        DataFrame(magicc_ci_ocean_heat_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "ci_ocean_heat.csv"),
        DataFrame(magicc_ci_oceanco2_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "ci_oceanco2_flux.csv"),
        DataFrame(indices=magicc_error_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "error_indices.csv"),
        DataFrame(indices=magicc_good_indices_wider) => joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "good_indices.csv"),
    ]
end


########################################################################
#----------------------------------------------------------------------#
#                           SC-CH4 Estimates                           #
#----------------------------------------------------------------------#
########################################################################

println("\nBEGIN ESTIMATING THE SC-CH4.\n")

#----------------------------------------------------------------------
# Load Data and Settings Common to All SC-CH4 Scenarios.
#----------------------------------------------------------------------

# Scenario settings.
pulse_year             = 2020
end_year               = 2300
dice_dollar_conversion = 1.06
fund_dollar_conversion = 1.35
low_ci_interval        = 0.95
high_ci_interval       = 0.98

@sync begin

    #---------------------------------------------------------------#
    #---------------------------------------------------------------#
    #------ Calculate SC-CH4 for Baseline Scenario (RCP 8.5). ------#
    #---------------------------------------------------------------#
    #---------------------------------------------------------------#



    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - DICE - FAIR" begin        
        # Load scenario temperature and CO₂ projections for each climate model.
        temperature_base_fairch4    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_temperature.csv"))))
        temperature_pulse_fairch4   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_temperature.csv"))))
        co2_base_fairch4            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_co2.csv"))))
        co2_pulse_fairch4           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4,   dice_error_indices_fairch4,   dice_good_indices_fairch4   = dice_damages(temperature_base_fairch4, temperature_pulse_fairch4, end_year)

        dice_scch4_fairch4_const25,   dice_discounted_damages_fairch4_const25     	= dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.025, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30     	= dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fairch4_const50,   dice_discounted_damages_fairch4_const50     	= dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.05, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fairch4_const70,   dice_discounted_damages_fairch4_const70     	= dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.07, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fairch4_ramsey_10, dice_discounted_damages_fairch4_ramsey_10     = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=false, η=1.0, ρ=0.015, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fairch4_ramsey_15, dice_discounted_damages_fairch4_ramsey_15     = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, end_year, constant=false, η=1.5, ρ=0.015, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_25.csv"),
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(scch4=dice_scch4_fairch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_50.csv"),
        DataFrame(scch4=dice_scch4_fairch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_70.csv"),
        DataFrame(scch4=dice_scch4_fairch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_ramsey_10.csv"),
        DataFrame(scch4=dice_scch4_fairch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_ramsey_15.csv"),
        DataFrame(dice_discounted_damages_fairch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "discounted_damages_25.csv"),
        DataFrame(dice_discounted_damages_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "discounted_damages_30.csv"),
        DataFrame(dice_discounted_damages_fairch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "discounted_damages_50.csv"),
        DataFrame(dice_discounted_damages_fairch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "discounted_damages_70.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - DICE - FUNDCH4" begin
        temperature_base_fundch4    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_temperature.csv"))))
        temperature_pulse_fundch4   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_temperature.csv"))))
        co2_base_fundch4            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_co2.csv"))))
        co2_pulse_fundch4           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4,   dice_error_indices_fundch4,   dice_good_indices_fundch4   = dice_damages(temperature_base_fundch4, temperature_pulse_fundch4, end_year)

        dice_scch4_fundch4_const25,   dice_discounted_damages_fundch4_const25     	= dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.025, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30     	= dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fundch4_const50,   dice_discounted_damages_fundch4_const50     	= dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.05, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fundch4_const70,   dice_discounted_damages_fundch4_const70     	= dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.07, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fundch4_ramsey_10, dice_discounted_damages_fundch4_ramsey_10     = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=false, η=1.0, ρ=0.015, dollar_conversion=dice_dollar_conversion)
        dice_scch4_fundch4_ramsey_15, dice_discounted_damages_fundch4_ramsey_15     = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, end_year, constant=false, η=1.5, ρ=0.015, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fundch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_25.csv"),
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(scch4=dice_scch4_fundch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_50.csv"),
        DataFrame(scch4=dice_scch4_fundch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_70.csv"),
        DataFrame(scch4=dice_scch4_fundch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_ramsey_10.csv"),
        DataFrame(scch4=dice_scch4_fundch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_ramsey_15.csv"),
        DataFrame(dice_discounted_damages_fundch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "discounted_damages_25.csv"),
        DataFrame(dice_discounted_damages_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "discounted_damages_30.csv"),
        DataFrame(dice_discounted_damages_fundch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "discounted_damages_50.csv"),
        DataFrame(dice_discounted_damages_fundch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "discounted_damages_70.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - DICE - HECTOR" begin

        temperature_base_hectorch4  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_temperature.csv"))))
        temperature_pulse_hectorch4 = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_temperature.csv"))))
        co2_base_hectorch4          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_co2.csv"))))
        co2_pulse_hectorch4         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(temperature_base_hectorch4, temperature_pulse_hectorch4, end_year)

        dice_scch4_hectorch4_const25,   dice_discounted_damages_hectorch4_const25 	= dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.025, dollar_conversion=dice_dollar_conversion)
        dice_scch4_hectorch4_const30,   dice_discounted_damages_hectorch4_const30   = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
        dice_scch4_hectorch4_const50,   dice_discounted_damages_hectorch4_const50   = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.05, dollar_conversion=dice_dollar_conversion)
        dice_scch4_hectorch4_const70,   dice_discounted_damages_hectorch4_const70   = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.07, dollar_conversion=dice_dollar_conversion)
        dice_scch4_hectorch4_ramsey_10, dice_discounted_damages_hectorch4_ramsey_10 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=false, η=1.0, ρ=0.015, dollar_conversion=dice_dollar_conversion)
        dice_scch4_hectorch4_ramsey_15, dice_discounted_damages_hectorch4_ramsey_15 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, end_year, constant=false, η=1.5, ρ=0.015, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_25.csv"),
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(scch4=dice_scch4_hectorch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_50.csv"),
        DataFrame(scch4=dice_scch4_hectorch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_70.csv"),
        DataFrame(scch4=dice_scch4_hectorch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_ramsey_10.csv"),
        DataFrame(scch4=dice_scch4_hectorch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_ramsey_15.csv"),
        DataFrame(dice_discounted_damages_hectorch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "discounted_damages_25.csv"),
        DataFrame(dice_discounted_damages_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "discounted_damages_30.csv"),
        DataFrame(dice_discounted_damages_hectorch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "discounted_damages_50.csv"),
        DataFrame(dice_discounted_damages_hectorch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "discounted_damages_70.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - DICE - MAGICC" begin
        temperature_base_magiccch4  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_temperature.csv"))))
        temperature_pulse_magiccch4 = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_temperature.csv"))))
        co2_base_magiccch4          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_co2.csv"))))
        co2_pulse_magiccch4         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(temperature_base_magiccch4, temperature_pulse_magiccch4, end_year)

        dice_scch4_magiccch4_const25,   dice_discounted_damages_magiccch4_const25 	= dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.025, dollar_conversion=dice_dollar_conversion)
        dice_scch4_magiccch4_const30,   dice_discounted_damages_magiccch4_const30 	= dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
        dice_scch4_magiccch4_const50,   dice_discounted_damages_magiccch4_const50 	= dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.05, dollar_conversion=dice_dollar_conversion)
        dice_scch4_magiccch4_const70,   dice_discounted_damages_magiccch4_const70   = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=true, η=0.0, ρ=0.07, dollar_conversion=dice_dollar_conversion)
        dice_scch4_magiccch4_ramsey_10, dice_discounted_damages_magiccch4_ramsey_10 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=false, η=1.0, ρ=0.015, dollar_conversion=dice_dollar_conversion)
        dice_scch4_magiccch4_ramsey_15, dice_discounted_damages_magiccch4_ramsey_15 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, end_year, constant=false, η=1.5, ρ=0.015, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_25.csv"),
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(scch4=dice_scch4_magiccch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_50.csv"),
        DataFrame(scch4=dice_scch4_magiccch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_70.csv"),
        DataFrame(scch4=dice_scch4_magiccch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_ramsey_10.csv"),
        DataFrame(scch4=dice_scch4_magiccch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_ramsey_15.csv"),
        DataFrame(dice_discounted_damages_magiccch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_25.csv"),
        DataFrame(dice_discounted_damages_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_30.csv"),
        DataFrame(dice_discounted_damages_magiccch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_50.csv"),
        DataFrame(dice_discounted_damages_magiccch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_70.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - FUND - FAIR" begin        
        # Load scenario temperature and CO₂ projections for each climate model.
        temperature_base_fairch4    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_temperature.csv"))))
        temperature_pulse_fairch4   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_temperature.csv"))))
        co2_base_fairch4            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "base_co2.csv"))))
        co2_pulse_fairch4           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(temperature_base_fairch4, co2_base_fairch4, temperature_pulse_fairch4, co2_pulse_fairch4, end_year)

        fund_scch4_fairch4_const25,   fund_discounted_damages_fairch4_const25       = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.025, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30       = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fairch4_const50,   fund_discounted_damages_fairch4_const50       = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.05, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fairch4_const70,   fund_discounted_damages_fairch4_const70       = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.07, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fairch4_ramsey_10, fund_discounted_damages_fairch4_ramsey_10     = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=false, η=1.0, γ=1.0, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fairch4_ramsey_15, fund_discounted_damages_fairch4_ramsey_15     = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, end_year, constant=false, η=1.5, γ=1.5, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_25.csv"),
        DataFrame(scch4=fund_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(scch4=fund_scch4_fairch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_50.csv"),
        DataFrame(scch4=fund_scch4_fairch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_70.csv"),
        DataFrame(scch4=fund_scch4_fairch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_ramsey_10.csv"),
        DataFrame(scch4=fund_scch4_fairch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_ramsey_15.csv"),
        DataFrame(fund_discounted_damages_fairch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "discounted_damages_25.csv"),
        DataFrame(fund_discounted_damages_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "discounted_damages_30.csv"),
        DataFrame(fund_discounted_damages_fairch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "discounted_damages_50.csv"),
        DataFrame(fund_discounted_damages_fairch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "discounted_damages_70.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - FUND - FUNDCH4" begin
        temperature_base_fundch4    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_temperature.csv"))))
        temperature_pulse_fundch4   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_temperature.csv"))))
        co2_base_fundch4            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "base_co2.csv"))))
        co2_pulse_fundch4           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(temperature_base_fundch4, co2_base_fundch4, temperature_pulse_fundch4, co2_pulse_fundch4, end_year)

        fund_scch4_fundch4_const25,   fund_discounted_damages_fundch4_const25       = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.025, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30       = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fundch4_const50,   fund_discounted_damages_fundch4_const50       = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.05, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fundch4_const70,   fund_discounted_damages_fundch4_const70       = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.07, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fundch4_ramsey_10, fund_discounted_damages_fundch4_ramsey_10     = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=false, η=1.0, γ=1.0, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_fundch4_ramsey_15, fund_discounted_damages_fundch4_ramsey_15     = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, end_year, constant=false, η=1.5, γ=1.5, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_25.csv"),
        DataFrame(scch4=fund_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(scch4=fund_scch4_fundch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_50.csv"),
        DataFrame(scch4=fund_scch4_fundch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_70.csv"),
        DataFrame(scch4=fund_scch4_fundch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_ramsey_10.csv"),
        DataFrame(scch4=fund_scch4_fundch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_ramsey_15.csv"),
        DataFrame(fund_discounted_damages_fundch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "discounted_damages_25.csv"),
        DataFrame(fund_discounted_damages_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "discounted_damages_30.csv"),
        DataFrame(fund_discounted_damages_fundch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "discounted_damages_50.csv"),
        DataFrame(fund_discounted_damages_fundch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "discounted_damages_70.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - FUND - HECTOR" begin
        temperature_base_hectorch4  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_temperature.csv"))))
        temperature_pulse_hectorch4 = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_temperature.csv"))))
        co2_base_hectorch4          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "base_co2.csv"))))
        co2_pulse_hectorch4         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(temperature_base_hectorch4, co2_base_hectorch4, temperature_pulse_hectorch4, co2_pulse_hectorch4, end_year)

        fund_scch4_hectorch4_const25,   fund_discounted_damages_hectorch4_const25   = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.025, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_hectorch4_const30,   fund_discounted_damages_hectorch4_const30   = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_hectorch4_const50,   fund_discounted_damages_hectorch4_const50   = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.05, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_hectorch4_const70,   fund_discounted_damages_hectorch4_const70   = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.07, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_hectorch4_ramsey_10, fund_discounted_damages_hectorch4_ramsey_10 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=false, η=1.0, γ=1.0, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_hectorch4_ramsey_15, fund_discounted_damages_hectorch4_ramsey_15 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, end_year, constant=false, η=1.5, γ=1.5, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_25.csv"),
        DataFrame(scch4=fund_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_30.csv"),
        DataFrame(scch4=fund_scch4_hectorch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_50.csv"),
        DataFrame(scch4=fund_scch4_hectorch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_70.csv"),
        DataFrame(scch4=fund_scch4_hectorch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_ramsey_10.csv"),
        DataFrame(scch4=fund_scch4_hectorch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_ramsey_15.csv"),
        DataFrame(fund_discounted_damages_hectorch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "discounted_damages_25.csv"),
        DataFrame(fund_discounted_damages_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "discounted_damages_30.csv"),
        DataFrame(fund_discounted_damages_hectorch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "discounted_damages_50.csv"),
        DataFrame(fund_discounted_damages_hectorch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "discounted_damages_70.csv"),
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Baseline Scenario (RCP 8.5) - FUND - MAGICC" begin
        temperature_base_magiccch4  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_temperature.csv"))))
        temperature_pulse_magiccch4 = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_temperature.csv"))))
        co2_base_magiccch4          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "base_co2.csv"))))
        co2_pulse_magiccch4         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "baseline_run", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(temperature_base_magiccch4, co2_base_magiccch4, temperature_pulse_magiccch4, co2_pulse_magiccch4, end_year)

        fund_scch4_magiccch4_const25,   fund_discounted_damages_magiccch4_const25 	= fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.025, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_magiccch4_const30,   fund_discounted_damages_magiccch4_const30 	= fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_magiccch4_const50,   fund_discounted_damages_magiccch4_const50 	= fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.05, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_magiccch4_const70,   fund_discounted_damages_magiccch4_const70 	= fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.07, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_magiccch4_ramsey_10, fund_discounted_damages_magiccch4_ramsey_10 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=false, η=1.0, γ=1.0, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
        fund_scch4_magiccch4_ramsey_15, fund_discounted_damages_magiccch4_ramsey_15 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=false, η=1.5, γ=1.5, ρ=0.015, dollar_conversion=fund_dollar_conversion, equity_weighting=false)

        fund_scch4_magiccch4_equity00 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.0, γ=0.0, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity01 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.1, γ=0.1, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity02 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.2, γ=0.2, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity03 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.3, γ=0.3, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity04 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.4, γ=0.4, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity05 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.5, γ=0.5, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity06 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.6, γ=0.6, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity07 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.7, γ=0.7, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity08 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.8, γ=0.8, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity09 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=0.9, γ=0.9, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity10 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1+1e-12, γ=1+1e-12, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity11 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.1, γ=1.1, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity12 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.2, γ=1.2, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity13 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.3, γ=1.3, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity14 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.4, γ=1.4, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity15 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.5, γ=1.5, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)
        fund_scch4_magiccch4_equity16 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, end_year, constant=true, η=1.6, γ=1.6, ρ=0.01, dollar_conversion=fund_dollar_conversion, equity_weighting=true)

        # Calculate regional confidence intervals for equity weighted SC-CH4 estimates across different η.
        fund_region_names = ["USA", "Canada", "Western Europe", "Japan and South Korea", "Australia and New Zealand", "Central and Eastern Europe", "Former Soviet Union", "Middle East", "Central America", "South America", "South Asia", "Southeast Asia", "China plus", "North Africa", "Sub Saharan Africa", "Small Island States"]

        fund_scch4_magiccch4_equity00_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity00, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity00_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity01_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity01, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity01_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity02_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity02, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity02_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity03_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity03, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity03_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity04_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity04, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity04_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity05_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity05, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity05_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity06_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity06, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity06_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity07_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity07, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity07_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity08_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity08, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity08_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity09_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity09, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity09_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity10_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity10, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity10_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity11_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity11, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity11_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity12_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity12, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity12_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity13_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity13, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity13_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity14_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity14, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity14_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity15_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity15, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity15_ci, :Year => :FUND_region)
        fund_scch4_magiccch4_equity16_ci = get_confidence_interval(fund_region_names, fund_scch4_magiccch4_equity16, low_ci_interval, high_ci_interval); rename!(fund_scch4_magiccch4_equity16_ci, :Year => :FUND_region)

        # FUND region names.
        col_names = Symbol.(["usa", "canada", "western_europe", "japan_south_korea", "australia_new_zealand", "central_eastern_europe", "former_soviet_union", "middle_east", "central_america", "south_america", "south_asia", "southeast_asia", "china_plus", "north_africa", "sub_saharan_africa", "small_island_states"])
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_25.csv"),
        DataFrame(scch4=fund_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(scch4=fund_scch4_magiccch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_50.csv"),
        DataFrame(scch4=fund_scch4_magiccch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_70.csv"),
        DataFrame(scch4=fund_scch4_magiccch4_ramsey_10) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_ramsey_10.csv"),
        DataFrame(scch4=fund_scch4_magiccch4_ramsey_15) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_ramsey_15.csv"),
        DataFrame(fund_discounted_damages_magiccch4_const25) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_25.csv"),
        DataFrame(fund_discounted_damages_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_30.csv"),
        DataFrame(fund_discounted_damages_magiccch4_const50) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_50.csv"),
        DataFrame(fund_discounted_damages_magiccch4_const70) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_70.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "baseline_run", "fund", "s_magicc", "good_indices.csv"),
        DataFrame(fund_scch4_magiccch4_equity00, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_00.csv"),
        DataFrame(fund_scch4_magiccch4_equity01, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_01.csv"),
        DataFrame(fund_scch4_magiccch4_equity02, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_02.csv"),
        DataFrame(fund_scch4_magiccch4_equity03, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_03.csv"),
        DataFrame(fund_scch4_magiccch4_equity04, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_04.csv"),
        DataFrame(fund_scch4_magiccch4_equity05, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_05.csv"),
        DataFrame(fund_scch4_magiccch4_equity06, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_06.csv"),
        DataFrame(fund_scch4_magiccch4_equity07, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_07.csv"),
        DataFrame(fund_scch4_magiccch4_equity08, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_08.csv"),
        DataFrame(fund_scch4_magiccch4_equity09, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_09.csv"),
        DataFrame(fund_scch4_magiccch4_equity10, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_10.csv"),
        DataFrame(fund_scch4_magiccch4_equity11, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_11.csv"),
        DataFrame(fund_scch4_magiccch4_equity12, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_12.csv"),
        DataFrame(fund_scch4_magiccch4_equity13, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_13.csv"),
        DataFrame(fund_scch4_magiccch4_equity14, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_14.csv"),
        DataFrame(fund_scch4_magiccch4_equity15, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_15.csv"),
        DataFrame(fund_scch4_magiccch4_equity16, Symbol.(col_names)) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_16.csv"),
        DataFrame(fund_scch4_magiccch4_equity00_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_00.csv"),
        DataFrame(fund_scch4_magiccch4_equity01_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_01.csv"),
        DataFrame(fund_scch4_magiccch4_equity02_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_02.csv"),
        DataFrame(fund_scch4_magiccch4_equity03_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_03.csv"),
        DataFrame(fund_scch4_magiccch4_equity04_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_04.csv"),
        DataFrame(fund_scch4_magiccch4_equity05_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_05.csv"),
        DataFrame(fund_scch4_magiccch4_equity06_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_06.csv"),
        DataFrame(fund_scch4_magiccch4_equity07_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_07.csv"),
        DataFrame(fund_scch4_magiccch4_equity08_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_08.csv"),
        DataFrame(fund_scch4_magiccch4_equity09_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_09.csv"),
        DataFrame(fund_scch4_magiccch4_equity10_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_10.csv"),
        DataFrame(fund_scch4_magiccch4_equity11_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_11.csv"),
        DataFrame(fund_scch4_magiccch4_equity12_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_12.csv"),
        DataFrame(fund_scch4_magiccch4_equity13_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_13.csv"),
        DataFrame(fund_scch4_magiccch4_equity14_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_14.csv"),
        DataFrame(fund_scch4_magiccch4_equity15_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_15.csv"),
        DataFrame(fund_scch4_magiccch4_equity16_ci) => joinpath(@__DIR__, output, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_16.csv"),
    ]

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Calculate SC-CH4 for RCP 2.6 Scenario.
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    @workunit "SC-CH4 for RCP 2.6 Scenario - DICE - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4,   dice_error_indices_fairch4,   dice_good_indices_fairch4   = dice_damages(fair_temperature_base, fair_temperature_pulse, 2300)

        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30   = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - DICE - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4,   dice_error_indices_fundch4,   dice_good_indices_fundch4   = dice_damages(fund_temperature_base, fund_temperature_pulse, 2300)

        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30   = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)        
    end [
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - DICE - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(hector_temperature_base, hector_temperature_pulse, 2300)        

        dice_scch4_hectorch4_const30, dice_discounted_damages_hectorch4_const30 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - DICE - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(magicc_temperature_base, magicc_temperature_pulse, 2300)        

        dice_scch4_magiccch4_const30, dice_discounted_damages_magiccch4_const30 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - FUND - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(fair_temperature_base, fair_co2_base, fair_temperature_pulse, fair_co2_pulse, 2300)

        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30   = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - FUND - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(fund_temperature_base, fund_co2_base, fund_temperature_pulse, fund_co2_pulse, 2300)

        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30   = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - FUND - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(hector_temperature_base, hector_co2_base, hector_temperature_pulse, hector_co2_pulse, 2300)

        fund_scch4_hectorch4_const30, fund_discounted_damages_hectorch4_const30 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_hector", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_hector", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for RCP 2.6 Scenario - FUND - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "rcp26", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(magicc_temperature_base, magicc_co2_base, magicc_temperature_pulse, magicc_co2_pulse, 2300)

        fund_scch4_magiccch4_const30, fund_discounted_damages_magiccch4_const30 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "rcp26", "fund", "s_magicc", "good_indices.csv"),
    ]

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Calculate SC-CH4 for Outdated CH₄ Forcing Scenario.
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------


    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - DICE - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4,   dice_error_indices_fairch4,   dice_good_indices_fairch4   = dice_damages(fair_temperature_base, fair_temperature_pulse, 2300)        

        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30   = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - DICE - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4,   dice_error_indices_fundch4,   dice_good_indices_fundch4   = dice_damages(fund_temperature_base, fund_temperature_pulse, 2300)

        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30   = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - DICE - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(hector_temperature_base, hector_temperature_pulse, 2300)

        dice_scch4_hectorch4_const30, dice_discounted_damages_hectorch4_const30 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - DICE - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(magicc_temperature_base, magicc_temperature_pulse, 2300)   
        
        dice_scch4_magiccch4_const30, dice_discounted_damages_magiccch4_const30 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - FUND - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(fair_temperature_base, fair_co2_base, fair_temperature_pulse, fair_co2_pulse, 2300)

        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30   = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - FUND - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(fund_temperature_base, fund_co2_base, fund_temperature_pulse, fund_co2_pulse, 2300)

        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30   = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - FUND - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(hector_temperature_base, hector_co2_base, hector_temperature_pulse, hector_co2_pulse, 2300)

        fund_scch4_hectorch4_const30, fund_discounted_damages_hectorch4_const30 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_hector", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_hector", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Outdated CH₄ Forcing Scenario - FUND - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "outdated_ch4_forcing", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(magicc_temperature_base, magicc_co2_base, magicc_temperature_pulse, magicc_co2_pulse, 2300)

        fund_scch4_magiccch4_const30, fund_discounted_damages_magiccch4_const30 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_magicc", "good_indices.csv"),
    ]


    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Calculate SC-CH4 for Scenario That Removes Posterior Parameter Correlations.
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - DICE - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4, dice_error_indices_fairch4, dice_good_indices_fairch4         = dice_damages(fair_temperature_base, fair_temperature_pulse, 2300)

        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30   = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - DICE - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4, dice_error_indices_fundch4, dice_good_indices_fundch4         = dice_damages(fund_temperature_base, fund_temperature_pulse, 2300)

        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30   = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - DICE - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(hector_temperature_base, hector_temperature_pulse, 2300)

        dice_scch4_hectorch4_const30, dice_discounted_damages_hectorch4_const30 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - DICE - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(magicc_temperature_base, magicc_temperature_pulse, 2300)

        dice_scch4_magiccch4_const30, dice_discounted_damages_magiccch4_const30 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - FUND - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(fair_temperature_base, fair_co2_base, fair_temperature_pulse, fair_co2_pulse, 2300)

        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30   = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)

        # Filter out small number of random parameter combinations for FUND that cause errors but don't crash the model (indicated by values much larger than ± $100,000)
        fund_non_extreme_indicies_fairch4   = findall(x-> abs(x) < 100000, fund_scch4_fairch4_const30)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const30[fund_non_extreme_indicies_fairch4]) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fair", "good_indices.csv"),
        DataFrame(indices=fund_non_extreme_indicies_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fair", "non_extreme_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - FUND - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(fund_temperature_base, fund_co2_base, fund_temperature_pulse, fund_co2_pulse, 2300)

        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30   = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)

        # Filter out small number of random parameter combinations for FUND that cause errors but don't crash the model (indicated by values much larger than ± $100,000)
        fund_non_extreme_indicies_fundch4   = findall(x-> abs(x) < 100000, fund_scch4_fundch4_const30)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const30[fund_non_extreme_indicies_fundch4]) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fund", "good_indices.csv"),
        DataFrame(indices=fund_non_extreme_indicies_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_fund", "non_extreme_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - FUND - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(hector_temperature_base, hector_co2_base, hector_temperature_pulse, hector_co2_pulse, 2300)

        fund_scch4_hectorch4_const30, fund_discounted_damages_hectorch4_const30 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)

        # Filter out small number of random parameter combinations for FUND that cause errors but don't crash the model (indicated by values much larger than ± $100,000)
        fund_non_extreme_indicies_hectorch4 = findall(x-> abs(x) < 100000, fund_scch4_hectorch4_const30)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const30[fund_non_extreme_indicies_hectorch4]) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_hector", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_hector", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_hector", "good_indices.csv"),
        DataFrame(indices=fund_non_extreme_indicies_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_hector", "non_extreme_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario That Removes Posterior Parameter Correlations - FUND - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "remove_correlations", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(magicc_temperature_base, magicc_co2_base, magicc_temperature_pulse, magicc_co2_pulse, 2300)

        fund_scch4_magiccch4_const30, fund_discounted_damages_magiccch4_const30 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)

        # Filter out small number of random parameter combinations for FUND that cause errors but don't crash the model (indicated by values much larger than ± $100,000)
        fund_non_extreme_indicies_magiccch4 = findall(x-> abs(x) < 100000, fund_scch4_magiccch4_const30)
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const30[fund_non_extreme_indicies_magiccch4]) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_magicc", "good_indices.csv"),
        DataFrame(indices=fund_non_extreme_indicies_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "remove_correlations", "fund", "s_magicc", "non_extreme_indices.csv"),
    ]

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Calculate .
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - DICE - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4,   dice_error_indices_fairch4,   dice_good_indices_fairch4   = dice_damages(fair_temperature_base, fair_temperature_pulse, 2300)

        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30   = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - DICE - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4,   dice_error_indices_fundch4,   dice_good_indices_fundch4   = dice_damages(fund_temperature_base, fund_temperature_pulse, 2300)

        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30   = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - DICE - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(hector_temperature_base, hector_temperature_pulse, 2300)

        dice_scch4_hectorch4_const30, dice_discounted_damages_hectorch4_const30 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - DICE - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(magicc_temperature_base, magicc_temperature_pulse, 2300)

        dice_scch4_magiccch4_const30, dice_discounted_damages_magiccch4_const30 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - FUND - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(fair_temperature_base, fair_co2_base, fair_temperature_pulse, fair_co2_pulse, 2300)

        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30   = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - FUND - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(fund_temperature_base, fund_co2_base, fund_temperature_pulse, fund_co2_pulse, 2300)

        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30   = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - FUND - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(hector_temperature_base, hector_co2_base, hector_temperature_pulse, hector_co2_pulse, 2300)

        fund_scch4_hectorch4_const30, fund_discounted_damages_hectorch4_const30 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "scch4_30.csv")
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "error_indices.csv")
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "good_indices.csv")
    ]

    @workunit "SC-CH4 for Scenario Using U.S. Climate Sensitivity Distribution - FUND - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "us_climate_sensitivity", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(magicc_temperature_base, magicc_co2_base, magicc_temperature_pulse, magicc_co2_pulse, 2300)

        fund_scch4_magiccch4_const30, fund_discounted_damages_magiccch4_const30 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "good_indices.csv"),
    ]

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    # Calculate SC-CH4 for Scenario Using Wider Prior Parameter Distributions.
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - DICE - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_co2.csv"))))

        dice_marginal_damages_fairch4,   dice_pc_consumption_fairch4,   dice_error_indices_fairch4,   dice_good_indices_fairch4   = dice_damages(fair_temperature_base, fair_temperature_pulse, 2300)

        dice_scch4_fairch4_const30,   dice_discounted_damages_fairch4_const30   = dice_scch4(dice_marginal_damages_fairch4, dice_pc_consumption_fairch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fair", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fair", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - DICE - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_co2.csv"))))

        dice_marginal_damages_fundch4,   dice_pc_consumption_fundch4,   dice_error_indices_fundch4,   dice_good_indices_fundch4   = dice_damages(fund_temperature_base, fund_temperature_pulse, 2300)

        dice_scch4_fundch4_const30,   dice_discounted_damages_fundch4_const30   = dice_scch4(dice_marginal_damages_fundch4, dice_pc_consumption_fundch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fund", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fund", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - DICE - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_co2.csv"))))

        dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, dice_error_indices_hectorch4, dice_good_indices_hectorch4 = dice_damages(hector_temperature_base, hector_temperature_pulse, 2300)

        dice_scch4_hectorch4_const30, dice_discounted_damages_hectorch4_const30 = dice_scch4(dice_marginal_damages_hectorch4, dice_pc_consumption_hectorch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_hector", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_hector", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - DICE - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_co2.csv"))))

        dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, dice_error_indices_magiccch4, dice_good_indices_magiccch4 = dice_damages(magicc_temperature_base, magicc_temperature_pulse, 2300)

        dice_scch4_magiccch4_const30, dice_discounted_damages_magiccch4_const30 = dice_scch4(dice_marginal_damages_magiccch4, dice_pc_consumption_magiccch4, pulse_year, 2300, constant=true, η=0.0, ρ=0.03, dollar_conversion=dice_dollar_conversion)
    end [
        DataFrame(scch4=dice_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=dice_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_magicc", "error_indices.csv"),
        DataFrame(indices=dice_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "dice", "s_magicc", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - FUND - FAIR" begin
        # Load temperature and CO₂ projections for each climate model.
        fair_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_temperature.csv"))))
        fair_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_temperature.csv"))))
        fair_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "base_co2.csv"))))
        fair_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fair", "pulse_co2.csv"))))

        fund_marginal_damages_fairch4,   fund_population_fairch4,   fund_consumption_fairch4,   fund_error_indices_fairch4,   fund_good_indices_fairch4   = fund_damages(fair_temperature_base, fair_co2_base, fair_temperature_pulse, fair_co2_pulse, 2300)

        fund_scch4_fairch4_const30,   fund_discounted_damages_fairch4_const30   = fund_scch4(fund_marginal_damages_fairch4, fund_consumption_fairch4, fund_population_fairch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fairch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fair", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fair", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fairch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fair", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - FUND - FUNDCH4" begin

        fund_temperature_base    = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_temperature.csv"))))
        fund_temperature_pulse   = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_temperature.csv"))))
        fund_co2_base            = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "base_co2.csv"))))
        fund_co2_pulse           = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_fund", "pulse_co2.csv"))))

        fund_marginal_damages_fundch4,   fund_population_fundch4,   fund_consumption_fundch4,   fund_error_indices_fundch4,   fund_good_indices_fundch4   = fund_damages(fund_temperature_base, fund_co2_base, fund_temperature_pulse, fund_co2_pulse, 2300)

        fund_scch4_fundch4_const30,   fund_discounted_damages_fundch4_const30   = fund_scch4(fund_marginal_damages_fundch4, fund_consumption_fundch4, fund_population_fundch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_fundch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fund", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fund", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_fundch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_fund", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - FUND - HECTOR" begin

        hector_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_temperature.csv"))))
        hector_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_temperature.csv"))))
        hector_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "base_co2.csv"))))
        hector_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_hector", "pulse_co2.csv"))))

        fund_marginal_damages_hectorch4, fund_population_hectorch4, fund_consumption_hectorch4, fund_error_indices_hectorch4, fund_good_indices_hectorch4 = fund_damages(hector_temperature_base, hector_co2_base, hector_temperature_pulse, hector_co2_pulse, 2300)

        fund_scch4_hectorch4_const30, fund_discounted_damages_hectorch4_const30 = fund_scch4(fund_marginal_damages_hectorch4, fund_consumption_hectorch4, fund_population_hectorch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_hectorch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_hector", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_hector", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_hectorch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_hector", "good_indices.csv"),
    ]

    @workunit "SC-CH4 for Scenario Using Wider Prior Parameter Distributions - FUND - MAGICC" begin

        magicc_temperature_base  = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_temperature.csv"))))
        magicc_temperature_pulse = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_temperature.csv"))))
        magicc_co2_base          = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "base_co2.csv"))))
        magicc_co2_pulse         = convert(Array{Float64,2}, DataFrame(load(joinpath(@__DIR__, output, "climate_projections", "wider_priors", "s_magicc", "pulse_co2.csv"))))

        fund_marginal_damages_magiccch4, fund_population_magiccch4, fund_consumption_magiccch4, fund_error_indices_magiccch4, fund_good_indices_magiccch4 = fund_damages(magicc_temperature_base, magicc_co2_base, magicc_temperature_pulse, magicc_co2_pulse, 2300)

        fund_scch4_magiccch4_const30, fund_discounted_damages_magiccch4_const30 = fund_scch4(fund_marginal_damages_magiccch4, fund_consumption_magiccch4, fund_population_magiccch4, pulse_year, 2300, constant=true, η=0.0, γ=0.0, ρ=0.03, dollar_conversion=fund_dollar_conversion, equity_weighting=false)
    end [
        DataFrame(scch4=fund_scch4_magiccch4_const30) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_magicc", "scch4_30.csv"),
        DataFrame(indices=fund_error_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_magicc", "error_indices.csv"),
        DataFrame(indices=fund_good_indices_magiccch4) => joinpath(@__DIR__, output, "scch4_estimates", "wider_priors", "fund", "s_magicc", "good_indices.csv"),
    ]

end

println("\nAll done.")
