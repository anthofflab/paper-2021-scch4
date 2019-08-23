
include("assim.jl")

include("calibration/sneasy_fund/assim.jl")
#include("calibration/sneasy_hector/assim_oldRF.jl")

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
    #strict_R_compat = false

    # Multiple chains? This was in the R version but doesn't currently work here
    #multiple_chains = false

    # Possible falues are :fortran and :julia
    #sneasy_version = :julia

    # Number of samples to draw during the MCMC
    #n_steps = 2_000_000

    # Number of samples to discard at the start (burn-in)
    #burn_in = 100_000

#######################################################################################################################################

function run_mcmc_h(mcmc_file::String, n_steps::Int64, burn_in::Int64, save_results::Bool, save_full_chain::Bool)

    run_mimi_sneasy! = construct_run_mimi_sneasy()

        log_post = construct_log_post(run_mimi_sneasy!)
    #else
    #    error("Unknown sneasy version requested")
    #end


    #--------------------------------------------------------------------------------------------------------
    # Maximize joint posterior pdf to find starting point for MCMC.
    #--------------------------------------------------------------------------------------------------------

    parnames =      ["S","kappa","alpha","Q10","beta","eta","T0","H0","CO20","sigma.T","sigma.H","sigma.CO2.inst","sigma.CO2.ice","rho.T","rho.H","rho.CO2.inst", "TOH0", "signma.ch4.inst", "rho.ch4.inst","signma.ch4.ice", "rho.ch4.ice", "CH40", "N2O0",  "F2x_CO2", "rf_scale_CH4"]
    initial_guess = [2.7,  2.9,    1.0,   4.2,   0.9,   23,-0.06,-33,   280,    0.1,     2.0,       0.45,              2.25,        0.55,   0.9,       0.95,       9.6,     4,                  0.5,            15.0,             0.5,     721.89411, 272.95961, 3.71,       1.0]

    #initial_guess = [2.7,2.9,1.0,4.2,0.9,23,-0.06,-33,285,0.1,2.0,0.45,2.25,0.55,0.9,0.95, 9.6, 4, 0.5,15.0, 0.5, 721.89411, 300.0, 120.0, 160.0, 272.95961]

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
    lifech4 = p[17]
    σ_ch4inst = p[18]
    ρ_ch4inst = p[19]
    σ_ch4ice = p[20]
    ρ_ch4ice = p[21]
    CH4_0 = p[22]
    N2O_0 = p[23]
    F2x_CO₂ = p[24]
    rf_scale_CH₄ = p[25]


    p0 = zeros(28)
    #if strict_R_compat
    	# These are the parameter estimates that R gets from maximising the posterior (ch4 parameters added without maximization here because Sneasy R does not have a CH4 cycle)
    #else
     #   println("Maximizing joint posterior pdf to find starting point for MCMC.")

        opt = Opt(:LN_NELDERMEAD, 28)
        ftol_rel!(opt, 1e-8)
        #lower_bounds!(opt, Float64[0,0,0,1,0,0,-Inf,-100,276,1e-10,1e-10,1e-10,1e-10,0,0,0,0,1e-10,0, 1e-10,0,696, 238, 96, 100, 265])
        #upper_bounds!(opt, Float64[Inf,Inf,3,5,1,200,Inf,0,281,0.2,4,1,10,0.99,0.99,0.99,20,20,0.99,30,0.99,747,484, 144, 200, 280])
        lower_bounds!(opt, Float64[0,0,0,1,0,0,-Inf,-100,276,1e-10,1e-10,1e-10,1e-10,0,0,0,0,1e-10,0, 1e-10,0,696, 238, 96, 100, 265, 2.968, 0.72])
        upper_bounds!(opt, Float64[Inf,Inf,3,5,1,200,Inf,0,281,0.2,4,1,10,0.99,0.99,0.99,20,20,0.99,30,0.99,747,484, 144, 200, 280,  4.452, 1.28])
        max_objective!(opt, (x, grad)->log_post(x))
    	maxtime!(opt, 360)

    	(minf,minx,ret) = optimize(opt, initial_guess)
    	p0 = minx
            mcmc_starting_point = minx

       # println("Finished maximizing joint posterior pdf.")
    #end


    #--------------------------------------------------------------------------------------------------------
    # Set up and run MCMC calibration.
    #--------------------------------------------------------------------------------------------------------


 mcmc_step = [0.16, 0.17, 0.025, 0.075, 0.015, 1, 0.003, 0.9, 0.5, 0.0005,0.025,0.0045,0.057,0.007,0.006,0.011,0.1,0.01,0.01,0.01,0.01,0.1,0.1, 0.01, 0.01]


    #job = BasicMCJob(model, RAM(diagm(step)), BasicMCRange(nsteps=n_steps, burnin=0), Dict(:p=>p0), verbose=true)
    #@time mcchain = run(job)

    function mcmc_stuff(n_steps::Int64, post_function, starting_vals, mcmc_step)
        model = likelihood_model(BasicContMuvParameter(:p, logtarget=post_function), false)
        job = BasicMCJob(model, RAM(diagm(mcmc_step)), BasicMCRange(nsteps=n_steps, burnin=Int64((0.25*n_steps))), Dict(:p=>starting_vals), verbose=false)
        mcchain = run(job)

        return output(job).value, mcchain
    end

chain, mcchain_object = mcmc_stuff(4_500_000, log_post, mcmc_starting_point, mcmc_step)



    model = likelihood_model(BasicContMuvParameter(:p, logtarget=log_post), false)

    step = [1.6,1.7,0.25,0.75,0.15,40,0.015,0.03,9,0.7,0.005,0.25,0.045,0.57,0.07,0.06,0.11,1.0,0.1,0.1,0.1,0.1,1.0,1.0,1.0,1.0]./10

    if multiple_chains
        job1 = BasicMCJob(model, MH(step), BasicMCRange(nsteps=10000, burnin=0), Dict(:p=>p0))
        @time mcchain1 = run(job1)

        job2 = BasicMCJob(model, MH(proposal_matrix(output(job1),mult=0.5)), BasicMCRange(nsteps=100_000, burnin=0), Dict(:p=>p0))
        @time mcchain2 = run(job2)

        job3 = BasicMCJob(model, MH(proposal_matrix(output(job2),mult=0.5)), BasicMCRange(nsteps=1_000_000, burnin=0), Dict(:p=>p0))
        @time mcchain3 = run(job3)

        job = job3
    else
        job = BasicMCJob(model, RAM(diagm(step)), BasicMCRange(nsteps=n_steps, burnin=0), Dict(:p=>p0), verbose=true)
        @time mcchain = run(job)
    end


    #--------------------------------------------------------------------------------------------------------
    # Save Results.
    #--------------------------------------------------------------------------------------------------------

    if save_results

        mcmc_file = "5_mill_cauchy_final_July6"
        # Create directory to save results
        output_folder = joinpath("calibration/sneasy_fund/mcmc_results", mcmc_file)
        mkdir(output_folder)

        # Extract mcmc results and discard burn-in samples.
        #chain = (output(job).value)[:, (burn_in+1):end]

        # Calculate mean value for each individual parameter in the chain.
        mean_chain = mean(chain, 2)
        median_chain = median(chain,2)
        mode_chain = zeros(length(mean_chain))
        for t = 1:length(mean_chain)
            mode_chain[t] = mode(chain[t,:])
            println(t)
        end
        
        # Create index to create thinned samples.
        thin_indices_500k = trunc(Int64, collect(linspace(1, size(chain,2), 500_000)))
        thin_indices_100k = trunc(Int64, collect(linspace(1, size(chain,2), 100_000)))
        thin_indices_10k = trunc(Int64, collect(linspace(1, size(chain,2), 10_000)))

        # Create thinned samples.
        thin_chain_500k = chain[:,thin_indices_500k];
        thin_chain_100k = chain[:,thin_indices_100k];
        thin_chain_10k = chain[:,thin_indices_10k];

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

    println("Calibration of SneasyCH4-H finished. Acceptance rate for burned sample = ", accept_rate)

end
