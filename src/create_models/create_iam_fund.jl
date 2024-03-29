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

    # Remove climate components.
    delete!(m, :climateco2cycle)
    delete!(m, :climatech4cycle)
    delete!(m, :climaten2ocycle)
    delete!(m, :climatesf6cycle)
    delete!(m, :climateforcing)
    delete!(m, :climatedynamics)

    # Set placeholder values for atmospheric CO₂ concentration.
    Mimi.set_external_param!(m, :acco2, zeros(n_steps), param_dims=[:time])
    connect_param!(m, :impactagriculture, :acco2, :acco2)
    connect_param!(m, :impactextratropicalstorms, :acco2, :acco2)
    connect_param!(m, :impactforests, :acco2, :acco2)

    # Set placeholder values for global temperature anomalies.
    Mimi.set_external_param!(m, :temp, zeros(n_steps), param_dims=[:time])
    connect_param!(m, :climateregional, :inputtemp, :temp)
    connect_param!(m, :biodiversity, :temp, :temp)
    connect_param!(m, :ocean, :temp, :temp)

    # Return modified version of FUND.
    return Mimi.build(m)
end
