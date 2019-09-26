
include("assim.jl")

include("calibration/sneasy_fair/assim.jl")
#include("calibration/sneasy_magicc/assim_oldRF.jl")

using NLopt
using Klara
using CSV
#######################################################################################################################################
# NOTES
#
# Set working directory to folder containing all Sneasy CH4 files: cd("My_Computer_Stuff.../sneasy_ch4")
#
# If save_file = true, the full mcmc results will be saved in the "mcmc_results" folder (along with thinned samples and the
# acceptance rate)
#
# Naming convention for mcmc file output is: "mcmc, number of samples, lenght of burn-in, sampler used"
#    e.g. "mcmc_1-000-000_10kburn_RAM.csv" takes one million samples with a burn-in of ten-thousand using the
#          RAM sampler . Thinned samples will have "thin_length".
#
########################################################################################################################################
# Things that may need to be changed depending on model run.

    # Should mcmc results be saved in a file
    #save_file = true

    # Should the full mcmc chain also be saved, or just the thinned versions?
    #save_full = false

    # If saving files, what is the file name
    #mcmc_file = "mcmc_2-000-000_100kburn_RAM"

    # Should the mcmc algorithm start with the same start values as the R code?
    strict_R_compat = false

    # Multiple chains? This was in the R version but doesn't currently work here
    multiple_chains = false

    # Possible falues are :fortran and :julia
    sneasy_version = :julia

    # Number of samples to draw during the MCMC
    #n_steps = 2_000_000

    # Number of samples to discard at the start (burn-in)
    #burn_in = 100_000

#######################################################################################################################################

function run_mcmc_m(mcmc_file::String, n_steps::Int64, burn_in::Int64, save_results::Bool, save_full_chain::Bool)

    run_mimi_sneasy! = construct_run_mimi_sneasy()

    # Get log-likelihood function
    log_post = construct_log_post(run_mimi_sneasy!)

    #--------------------------------------------------------------------------------------------------------
    # Maximize joint posterior pdf to find starting point for MCMC.
    #--------------------------------------------------------------------------------------------------------

    parnames = ["S","kappa","alpha","Q10","beta","eta","T0","H0","CO20","sigma.T","sigma.H","sigma.CO2.inst","sigma.CO2.ice","rho.T","rho.H","rho.CO2.inst", "tau", "signma.ch4.inst", "rho.ch4.inst","signma.ch4.ice", "rho.ch4.ice", "CH4_1", "CH4_2", "CH4_Nat", "T_Soil", "T_Strat", "N2O_0", "F2x_CO₂", "rf_scale_CH₄"]

    #initial_guess = [2.7,2.9,1.0,4.2,0.9,23,-0.06,-33,286,0.1,2.0,0.45,2.25,0.55,0.9,0.95, 9.6, 4, 0.5,15.0, 0.5, 722.0, 300.0, 120.0, 160.0, 272.95961]
    initial_guess = [2.7,2.9,1.0,4.2,0.9,23,-0.06,-33,280,0.1,2.0,0.45,2.25,0.55,0.9,0.95, 9.6, 4, 0.5,15.0, 0.5, 721.89411, 300.0, 272.95961, 3.71, 1.0]

    p0 = zeros(26)
    
    println("Maximizing joint posterior pdf to find starting point for MCMC.")

 	opt = Opt(:LN_NELDERMEAD, 26)
    ftol_rel!(opt, 1e-8)
 	#lower_bounds!(opt, Float64[0,0,0,1,0,0,-Inf,-100,280,1e-10,1e-10,1e-10,1e-10,0,0,0,0,1e-10,0, 1e-10,0,697, 238, 96, 100, 266])
   	#upper_bounds!(opt, Float64[Inf,Inf,3,5,1,200,Inf,0,295,0.2,4,1,10,0.99,0.99,0.99,20,20,0.99,30,0.99,747,484, 144, 200, 280])
   	lower_bounds!(opt, Float64[0,0,0,1,0,0,-Inf,-100,276,1e-10,1e-10,1e-10,1e-10,0,0,0,0,1e-10,0, 1e-10,0,696, 150, 265, 2.968, 0.72])
    upper_bounds!(opt, Float64[Inf,Inf,3,5,1,200,Inf,0,281,0.2,4,1,10,0.99,0.99,0.99,20,20,0.99,30,0.99,747,484, 280, 4.452, 1.28])
    max_objective!(opt, (x, grad)->log_post(x))
   	maxtime!(opt, 300)

   	(minf,minx,ret) = optimize(opt, initial_guess)
   	mcmc_starting_point = minx
    println("Finished maximizing joint posterior pdf.")


    #--------------------------------------------------------------------------------------------------------
    # Set up and run MCMC calibration.
    #--------------------------------------------------------------------------------------------------------

    model = likelihood_model(BasicContMuvParameter(:p, logtarget=log_post), false)

    mcmc_step = [0.16, 0.17, 0.025, 0.075, 0.015, 1, 0.003, 0.9, 0.5, 0.0005,0.025,0.0045,0.057,0.007,0.006,0.011,0.1,0.01,0.01,0.01,0.01,0.1,0.1, 0.1, 0.01, 0.01]


    #job = BasicMCJob(model, RAM(diagm(step)), BasicMCRange(nsteps=n_steps, burnin=0), Dict(:p=>p0), verbose=true)
    #@time mcchain = run(job)

    function mcmc_stuff(n_steps::Int64, post_function, starting_vals, mcmc_step)
        model = likelihood_model(BasicContMuvParameter(:p, logtarget=post_function), false)
        job = BasicMCJob(model, RAM(diagm(mcmc_step)), BasicMCRange(nsteps=n_steps, burnin=Int64((0.1*n_steps))), Dict(:p=>starting_vals), verbose=false)
        mcchain = run(job)

        return output(job).value, mcchain
    end

chain, mcchain_object = mcmc_stuff(5_000_000, log_post, mcmc_starting_point, mcmc_step)

# Second attempt
chain_2, mcchain_object_2 = mcmc_stuff(5_000_000, log_post, mcmc_starting_point_2, mcmc_step)
    mcmc_starting_point_2 = [3.5, 2.5, 1.5, 3.0,    0.5,   50.0,  0.01,  -50.0, 277.0,   0.15,       3.0,        0.5,          5.0,        0.5,      0.5,          0.5,        7.1,     10.5,          0.5,          20.0,      0.5,        700.0,    200.0,     280.0,     3.6,           1.1]
                             "S", "κ", "α", "Q10", "beta", "eta", "T0", "H0", "CO20", "σ_temp", "σ_ocheat",  "σ_co2inst", "σ_co2ice",  "ρ_temp",  "ρ_ocheat",  "ρ_co2inst",  "τ",  "σ_ch4inst",  "ρ_ch4inst",  "σ_ch4ice",  "ρ_ch4ice",  "CH4_0",  "CH4_Nat",  "N2O_0",  "F2x_CO₂",  "rf_scale_CH₄",   #--------------------------------------------------------------------------------------------------------
       # Save Results.
    #--------------------------------------------------------------------------------------------------------

    if save_results

        # Create directory to save results
        output_folder = joinpath("calibration/sneasy_fair/mcmc_results", mcmc_file)
        mkdir(output_folder)

        # Extract mcmc results and discard burn-in samples.
        #chain = (output(job).value)[:, (burn_in+1):end]

        # Calculate mean value for each individual parameter in the chain.
        mean_chain = mean(chain_2, 2)
        median_chain = median(chain_2,2)
        mode_chain = zeros(26)
        for t = 1:26
            mode_chain[t] = mode(chain_2[t,:])
            println(t)
        end
        # Create index to create thinned samples.
        thin_indices_500k = trunc(Int64, collect(linspace(1, size(chain_2,2), 500_000)))
        thin_indices_100k = trunc(Int64, collect(linspace(1, size(chain_2,2), 100_000)))
        thin_indices_10k = trunc(Int64, collect(linspace(1, size(chain_2,2), 10_000)))

        # Create thinned samples.
        thin_chain_500k = chain_2[:,thin_indices_500k];
        thin_chain_100k = chain_2[:,thin_indices_100k];
        thin_chain_10k = chain_2[:,thin_indices_10k];

        # Caluclate acceptance rate.
        accept_rate = mcmc_acceptance(chain[1,:])

        # Create files
        CSVFiles.save(joinpath(output_folder, "thin_500k.csv"), header=true, DataFrame(transpose(thin_chain_500k)))
        CSVFiles.save(joinpath(output_folder, "thin_100k.csv"), header=true, DataFrame(transpose(thin_chain_100k)))
        CSVFiles.save(joinpath(output_folder, "thin_10k.csv"), header=true, DataFrame(transpose(thin_chain_10k)))
        #CSVFiles.save(joinpath(output_folder, "maxpost_params.csv"), header=true, DataFrame(max_post_params = p0))
        CSVFiles.save(joinpath(output_folder, "mean_params.csv"), header=true, DataFrame(mean=mean_chain[:]))
        CSVFiles.save(joinpath(output_folder, "median_params.csv"), header=true, DataFrame(median=median_chain[:]))
        CSVFiles.save(joinpath(output_folder, "mode_params.csv"), header=true, DataFrame(mode=mode_chain[:]))

        CSVFiles.save(joinpath(output_folder, "acceptance.csv"), header=true, DataFrame(accept_rate=accept_rate))
        CSVFiles.save(joinpath(output_folder, "parameter_names.csv"), header=true, DataFrame(names=parnames))

        # Save the full chain (will be a very large file).
        if save_full_chain
            writetable(joinpath(output_folder, "full.csv"), header=true, DataFrame(chain))
        end

    end

    println("Calibration of SneasyCH4-M finished. Acceptance rate for burned sample = ", accept_rate)

end
