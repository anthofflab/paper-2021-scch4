#----------------------------------------------------------------------------------------
# This function creates an instance of DICE used to calculate the SC-CH4 with SNEASY+CH4.
#----------------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiDICE2013


function create_iam_dice(;end_year::Int=2300)

    # Create an instance of DICE2013.
    m = MimiDICE2013.get_model()

    # Set model time horizon (note DICE runs on 5-year timesteps).
    set_dimension!(m, :time, 2010:5:end_year)

    # Calculate default number of model timesteps.
    n_steps = length(dim_keys(m, :time))

    # Remove climate and welfare components.
    delete!(m, :emissions)
    delete!(m, :co2cycle)
    delete!(m, :radiativeforcing)
    delete!(m, :climatedynamics)
    delete!(m, :welfare)

    # Set a placeholder for temperature anomaly in DICE2013 damage function.
    set_param!(m, :damages, :TATM, zeros(n_steps))

    # Return modified version of DICE2013.
    return m
end
