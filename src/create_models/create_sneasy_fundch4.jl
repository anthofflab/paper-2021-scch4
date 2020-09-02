#-------------------------------------------------------------------------------
# This function creates an instance of SNEASY+FUND.
#-------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiFUND
using MimiSNEASY
using DataFrames
using CSVFiles

# Load required files.
include(joinpath("additional_model_components", "rf_total.jl"))
include(joinpath("additional_model_components", "rf_co2_etminan.jl"))
include(joinpath("additional_model_components", "rf_ch4_etminan.jl"))
include(joinpath("additional_model_components", "rf_ch4_total_fund.jl"))
include(joinpath("additional_model_components", "rf_ch4_direct_fund.jl"))
include(joinpath("..", "helper_functions.jl"))

function create_sneasy_fundch4(;rcp_scenario::String="RCP85", start_year::Int=1765, end_year::Int=2300, etminan_ch4_forcing::Bool=true)

    # ---------------------------------------------
    # Load and clean up necessary data.
    # ---------------------------------------------

    # Load RCP emissions and concentration scenario values (RCP options = "RCP26" and "RCP85").
    rcp_emissions      = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_emissions.csv"), skiplines_begin=36))
    rcp_concentrations = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_concentrations.csv"), skiplines_begin=37))
    rcp_forcing        = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_midyear_radforcings.csv"), skiplines_begin=58))

    # Load FUND data for RCP non-methane tropospheric ozone radiative forcing.
    fund_non_CH₄_O₃forcing = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_nonch4_tropo3_forcing_fund.csv")))

    # Find start and end year indices to crop RCP scenario data to correct model time horizon.
    rcp_indices = findall((in)(collect(start_year:end_year)), rcp_emissions.YEARS)

    # Set pre-industrial atmospheric CO₂, CH₄, and N₂O concentrations to RCP values in 1765.
    CO₂_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CO2][1]
    CH₄_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CH4][1]
    N₂O_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :N2O][1]

    # Calculate carbon dioxide emissions as well as aerosol and exogenous RCP radiative forcing scenarios.
    rcp_co2_emissions     = rcp_emissions.FossilCO2 .+ rcp_emissions.OtherCO2
    rcp_aerosol_forcing   = rcp_forcing.TOTAER_DIR_RF .+ rcp_forcing.CLOUD_TOT_RF
    rcp_exogenous_forcing = rcp_forcing.TOTAL_INCLVOLCANIC_RF .- rcp_forcing.CO2_RF .- rcp_forcing.CH4_RF .- rcp_forcing.TROPOZ_RF .+ fund_non_CH₄_O₃forcing.nonch4_forcing .- rcp_forcing.CH4OXSTRATH2O_RF .- rcp_aerosol_forcing


    # ------------------------------------------------------------
    # Initialize Mimi-SNEASY and add new CH₄ components to model.
    # ------------------------------------------------------------

    # Get an instance of Mimi-SNEASY.
    m = MimiSNEASY.get_model(start_year=start_year, end_year=end_year)

    # Remove old radiative forcing components.
    delete!(m, :rfco2)
    delete!(m, :radiativeforcing)

    # Add in new components.
    add_comp!(m, rf_total, before = :doeclim)
    add_comp!(m, rf_co2_etminan, before = :rf_total)
    add_comp!(m, rf_ch4_total_fund, before = :rf_co2_etminan)
    add_comp!(m, MimiFUND.climatech4cycle , before= :rf_ch4_total_fund)

    # Add in user-specified CH₄ radiative forcing component.
    # Note: If not using Etminan et al. equations, use original forcing equations from parent CH₄ model.
    if etminan_ch4_forcing == true
        add_comp!(m, rf_ch4_etminan, before = :rf_ch4_total_fund)
    else
        add_comp!(m, rf_ch4_direct_fund, before = :rf_ch4_total_fund)
    end


    # ---------------------------------------------
    # Set component parameters.
    # ---------------------------------------------

    # ---- Common parameters ---
    Mimi.set_external_param!(m, :CH₄_0, CH₄_0)
    Mimi.set_external_param!(m, :N₂O_0, N₂O_0)
    Mimi.set_external_param!(m, :N₂O, rcp_concentrations.N2O[rcp_indices])

    # ---- Carbon Cycle ---- #
    update_param!(m, :CO2_emissions, rcp_co2_emissions[rcp_indices])
    update_param!(m, :atmco20, CO₂_0)

    # ---- Methane Cycle ---- #
    set_param!(m, :climatech4cycle, :lifech4, 12.0)
    set_param!(m, :climatech4cycle, :ch4pre, CH₄_0)
    set_param!(m, :climatech4cycle, :globch4, rcp_emissions.CH4[rcp_indices])
    set_param!(m, :climatech4cycle, :acch4_0, CH₄_0)

    # ---- Direct Methane Radiative Forcing ---- #
    if etminan_ch4_forcing == true
        connect_param!(m, :rf_ch4_etminan, :CH₄_0, :CH₄_0)
        connect_param!(m, :rf_ch4_etminan, :N₂O_0, :N₂O_0)
        set_param!(m, :rf_ch4_etminan, :scale_CH₄, 1.0)
        set_param!(m, :rf_ch4_etminan, :a₃, -1.3e-6)
        set_param!(m, :rf_ch4_etminan, :b₃, -8.2e-6)
        connect_param!(m, :rf_ch4_etminan, :N₂O, :N₂O)
    else
        connect_param!(m, :rf_ch4_direct_fund, :N₂O_0, :N₂O_0)
        connect_param!(m, :rf_ch4_direct_fund, :CH₄_0, :CH₄_0)
        set_param!(m, :rf_ch4_direct_fund, :scale_CH₄, 1.0)
    end

    # ---- Total Methane Radiative Forcing (including indirect effects) ---- #
    connect_param!(m, :rf_ch4_total_fund, :CH₄_0, :CH₄_0)
    set_param!(m, :rf_ch4_total_fund, :ϕ, 0.4)

    # ---- Carbon Dioxide Radiative Forcing ---- #
    set_param!(m, :rf_co2_etminan, :a₁, -2.4e-7)
    set_param!(m, :rf_co2_etminan, :b₁, 7.2e-4)
    set_param!(m, :rf_co2_etminan, :c₁, -2.1e-4)
    set_param!(m, :rf_co2_etminan, :CO₂_0, CO₂_0)
    connect_param!(m, :rf_co2_etminan, :N₂O_0, :N₂O_0)
    connect_param!(m, :rf_co2_etminan, :N₂O, :N₂O)
    set_param!(m, :rf_co2_etminan, :rf_scale_CO₂, co2_rf_scale(3.7, CO₂_0, N₂O_0))

    # ---- Total Radiative Forcing ---- #
    set_param!(m, :rf_total, :α, 1.0)
    # TODO It would be nice if `rf_aerosol` didn't exist as an external parameter
    # at this point
    connect_param!(m, :rf_total, :rf_aerosol, :rf_aerosol)
    update_param!(m, :rf_aerosol, rcp_aerosol_forcing[rcp_indices])
    set_param!(m, :rf_total, :rf_exogenous, rcp_exogenous_forcing[rcp_indices])
    set_param!(m, :rf_total, :rf_O₃, zeros(length(start_year:end_year)))
    set_param!(m, :rf_total, :rf_CH₄_H₂O, zeros(length(start_year:end_year)))    

    # ----------------------------------------------------------
    # Create connections between Mimi SNEASY+Hector components.
    # ----------------------------------------------------------
    connect_param!(m, :doeclim,           :forcing, :rf_total,        :total_forcing)
    connect_param!(m, :ccm,               :temp,    :doeclim,         :temp)
    connect_param!(m, :rf_co2_etminan,    :CO₂,     :ccm,             :atmco2)
    connect_param!(m, :rf_ch4_total_fund, :CH₄,     :climatech4cycle, :acch4)

    # Create different connections if using updated Etminan et al. CH₄ forcing equations.
    if etminan_ch4_forcing == true
        connect_param!(m, :rf_ch4_etminan,     :CH₄,           :climatech4cycle,    :acch4)
        connect_param!(m, :rf_ch4_total_fund,  :rf_ch4_direct, :rf_ch4_etminan,     :rf_CH₄)
    else
        connect_param!(m, :rf_ch4_direct_fund, :CH₄,           :climatech4cycle,    :acch4)
        connect_param!(m, :rf_ch4_total_fund,  :rf_ch4_direct, :rf_ch4_direct_fund, :rf_ch4_direct)
    end

    connect_param!(m, :rf_total, :rf_CH₄, :rf_ch4_total_fund, :rf_ch4_total)
    connect_param!(m, :rf_total, :rf_CO₂, :rf_co2_etminan,    :rf_CO₂)

    # Return constructed model.
    return m
end
