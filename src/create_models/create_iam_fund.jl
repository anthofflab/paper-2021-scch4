#----------------------------------------------------------------------------------------
# This function creates an instance of FUND used to calculate the SC-CH4 with SNEASY+CH4.
#----------------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiFUND


function create_iam_fund(;end_year::Int=2300)

    # Get number of timesteps to run model for.
    n_steps = length(1950:end_year)

    # Create an instance of Mimi-FUND (time dimension will be set based on nsteps and offset by 1).
    m = MimiFUND.get_model(nsteps=n_steps-1)

    # Remove climate and welfare components.
    delete!(m, :climateco2cycle)
    delete!(m, :climatech4cycle)
    delete!(m, :climaten2ocycle)
    delete!(m, :climatesf6cycle)
    delete!(m, :climateforcing)
    delete!(m, :climatedynamics)

    # Set placeholder values for atmospheric CO₂ concentration.
    set_param!(m, :impactagriculture, :acco2, zeros(n_steps))
    set_param!(m, :impactextratropicalstorms, :acco2, zeros(n_steps))
    set_param!(m, :impactforests, :acco2, zeros(n_steps))

    # Set placeholder values for global temperature anomalies.
    set_param!(m, :climateregional, :inputtemp, zeros(n_steps))
    set_param!(m, :biodiversity, :temp, zeros(n_steps))
    set_param!(m, :ocean, :temp, zeros(n_steps))

    # Return modified version of FUND.
    return m
end
