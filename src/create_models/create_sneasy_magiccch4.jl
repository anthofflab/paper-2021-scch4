#-------------------------------------------------------------------------------
# This function creates an instance of SNEASY+MAGICC.
#-------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiMAGICC
using MimiSNEASY
using DataFrames
using CSVFiles

# Load required files.
include(joinpath("additional_model_components", "rf_total.jl"))
include(joinpath("additional_model_components", "rf_co2_etminan.jl"))
include(joinpath("additional_model_components", "rf_ch4_etminan.jl"))
include(joinpath("additional_model_components", "total_co2_emissions.jl"))
include(joinpath("..", "helper_functions.jl"))

function create_sneasy_magiccch4(;rcp_scenario::String="RCP85", start_year::Int=1765, end_year::Int=2300, etminan_ch4_forcing::Bool=true)

    # ---------------------------------------------
    # Load and clean up necessary data.
    # ---------------------------------------------

    # Load RCP emissions and concentration scenario values (RCP options = "RCP26" and "RCP85").
    rcp_emissions      = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_emissions.csv"), skiplines_begin=36))
    rcp_concentrations = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_concentrations.csv"), skiplines_begin=37))
    rcp_forcing        = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_midyear_radforcings.csv"), skiplines_begin=58))

    # Load MAGICC data for historical fossil carbon dioxide emissions values used to calculate pre-2000 non-methane ozone forcing.
    foss_hist_for_O₃   = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", "FOSS_HIST_magicc.csv")))

    # Find start and end year indices to crop RCP scenario data to correct model time horizon.
    rcp_indices = findall((in)(collect(start_year:end_year)), rcp_emissions.YEARS)

    # Set pre-industrial atmospheric CO₂, CH₄, and N₂O concentrations to RCP values in 1765.
    CO₂_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CO2][1]
    CH₄_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CH4][1]
    N₂O_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :N2O][1]

    # Calculate carbon dioxide emissions as well as aerosol and exogenous RCP radiative forcing scenarios.
    rcp_co2_emissions     = rcp_emissions.FossilCO2 .+ rcp_emissions.OtherCO2
    rcp_aerosol_forcing   = rcp_forcing.TOTAER_DIR_RF .+ rcp_forcing.CLOUD_TOT_RF
    rcp_exogenous_forcing = rcp_forcing.TOTAL_INCLVOLCANIC_RF .- rcp_forcing.CO2_RF .- rcp_forcing.CH4_RF .- rcp_forcing.TROPOZ_RF .- rcp_forcing.CH4OXSTRATH2O_RF .- rcp_aerosol_forcing


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
    add_comp!(m, MimiMAGICC.rf_ch4h2o, before = :rf_co2_etminan)
    add_comp!(m, MimiMAGICC.rf_o3, before = :rf_ch4h2o)
    add_comp!(m, total_co2_emissions, before= :rf_o3)
    add_comp!(m, MimiMAGICC.ch4_cycle, before = :total_co2_emissions)

    # Add in user-specified CH₄ radiative forcing component.
    # Note: If not using Etminan et al. equations, use original forcing equations from parent CH₄ model.
    if etminan_ch4_forcing == true
        add_comp!(m, rf_ch4_etminan, before = :rf_ch4h2o)
    else
        add_comp!(m, MimiMAGICC.rf_ch4, before = :rf_ch4h2o)
    end


    # ---------------------------------------------
    # Set component parameters.
    # ---------------------------------------------

    # ---- Common parameters ----
    Mimi.set_external_param!(m, :CH₄_0, CH₄_0)
    Mimi.set_external_param!(m, :N₂O_0, N₂O_0)
    Mimi.set_external_param!(m, :N₂O, rcp_concentrations.N2O[rcp_indices])
    Mimi.set_external_param!(m, :CO_emissions, rcp_emissions.CO[rcp_indices])
    Mimi.set_external_param!(m, :NMVOC_emissions, rcp_emissions.NMVOC[rcp_indices])
    Mimi.set_external_param!(m, :NOx_emissions, rcp_emissions.NOx[rcp_indices])    

    # ---- Carbon Cycle ---- #
    update_param!(m, :atmco20, CO₂_0)

    # ---- Methane Cycle ---- #
    set_param!(m, :ch4_cycle, :BBCH4, 2.78)
    set_param!(m, :ch4_cycle, :ANOX, 0.0042)
    set_param!(m, :ch4_cycle, :ACO, -0.000105)
    set_param!(m, :ch4_cycle, :AVOC, -0.000315)
    set_param!(m, :ch4_cycle, :TAUINIT, 7.73)
    set_param!(m, :ch4_cycle, :SCH4, -0.32)
    set_param!(m, :ch4_cycle, :TAUSOIL, 160.)
    set_param!(m, :ch4_cycle, :TAUSTRAT, 120.)
    set_param!(m, :ch4_cycle, :GAM, -1.0)
    set_param!(m, :ch4_cycle, :CH4_natural, 266.5)
    set_param!(m, :ch4_cycle, :fffrac, 0.18)
    connect_param!(m, :ch4_cycle, :CH₄_0, :CH₄_0)
    set_param!(m, :ch4_cycle, :CH4_emissions, rcp_emissions.CH4[rcp_indices])
    connect_param!(m, :ch4_cycle, :NOX_emissions, :NOX_emissions)
    connect_param!(m, :ch4_cycle, :CO_emissions, :CO_emissions)
    connect_param!(m, :ch4_cycle, :NMVOC_emissions, :NMVOC_emissions)

    # ---- Total Carbon Dioxide Emissions ---- #
    set_param!(m, :total_co2_emissions, :exogenous_CO₂_emissions, rcp_co2_emissions[rcp_indices])

    # ---- Methane Radiative Forcing ---- #
    if etminan_ch4_forcing == true
        connect_param!(m, :rf_ch4_etminan, :CH₄_0, :CH₄_0)
        connect_param!(m, :rf_ch4_etminan, :N₂O_0, :N₂O_0)        
        set_param!(m, :rf_ch4_etminan, :scale_CH₄, 1.0)
        set_param!(m, :rf_ch4_etminan, :a₃, -1.3e-6)
        set_param!(m, :rf_ch4_etminan, :b₃, -8.2e-6)
        connect_param!(m, :rf_ch4_etminan, :N₂O, :N₂O)
    else
        connect_param!(m, :rf_ch4, :N₂O_0, :N₂O_0)
        connect_param!(m, :rf_ch4, :CH₄_0, :CH₄_0)
        set_param!(m, :rf_ch4, :scale_CH₄, 1.0)
    end

    # ---- Straospheric Water Vapor From Oxidized Methane Radiative Forcing ---- #
    connect_param!(m, :rf_ch4h2o, :CH₄_0, :CH₄_0)
    set_param!(m, :rf_ch4h2o, :STRATH2O, 0.15)

    # ---- Tropospheric Ozone Radiative Forcing ---- #
    connect_param!(m, :rf_o3, :CH₄_0, :CH₄_0)
    set_param!(m, :rf_o3, :OZ00CH4, 0.161)
    set_param!(m, :rf_o3, :OZNOX, 0.125)
    set_param!(m, :rf_o3, :OZCO, 0.0011)
    set_param!(m, :rf_o3, :OZVOC, 0.0033)
    connect_param!(m, :rf_o3, :NOx_emissions, :NOx_emissions)
    connect_param!(m, :rf_o3, :CO_emissions, :CO_emissions)
    connect_param!(m, :rf_o3, :NMVOC_emissions, :NMVOC_emissions)    
    set_param!(m, :rf_o3, :FOSSHIST, foss_hist_for_O₃.EFOSS[rcp_indices])
    set_param!(m, :rf_o3, :TROZSENS, 0.042)
    set_param!(m, :rf_o3, :OZCH4, 5.0)

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
    # TODO It would be nice if `rf_aerosol` would not be a model
    # parameter at this point
    connect_param!(m, :rf_total, :rf_aerosol, :rf_aerosol)
    update_param!(m, :rf_aerosol, rcp_aerosol_forcing[rcp_indices])
    set_param!(m, :rf_total, :rf_exogenous, rcp_exogenous_forcing[rcp_indices])

    # ----------------------------------------------------------
    # Create connections between Mimi SNEASY+Hector components.
    # ----------------------------------------------------------
    connect_param!(m, :doeclim,             :forcing,             :rf_total,            :total_forcing)
    connect_param!(m, :ch4_cycle,           :temperature,         :doeclim,             :temp)
    connect_param!(m, :total_co2_emissions, :oxidized_CH₄_to_CO₂, :ch4_cycle,           :emeth)
    connect_param!(m, :ccm,                 :CO2_emissions,       :total_co2_emissions, :total_CO₂_emissions)
    connect_param!(m, :ccm,                 :temp,                :doeclim,             :temp)
    connect_param!(m, :rf_co2_etminan,      :CO₂,                 :ccm,                 :atmco2)
    connect_param!(m, :rf_o3,               :CH₄,                 :ch4_cycle,           :CH₄)
    connect_param!(m, :rf_ch4h2o,           :CH₄,                 :ch4_cycle,           :CH₄)

    # Create different connections if using updated Etminan et al. CH₄ forcing equations.
    if etminan_ch4_forcing == true
        connect_param!(m, :rf_ch4_etminan, :CH₄,    :ch4_cycle,      :CH₄)
        connect_param!(m, :rf_total,       :rf_CH₄, :rf_ch4_etminan, :rf_CH₄)
    else
        connect_param!(m, :rf_ch4,         :CH₄,    :ch4_cycle,      :CH₄)
        connect_param!(m, :rf_total,       :rf_CH₄, :rf_ch4,         :QMeth)
    end

    connect_param!(m, :rf_total, :rf_O₃,      :rf_o3,          :rf_O₃)
    connect_param!(m, :rf_total, :rf_CH₄_H₂O, :rf_ch4h2o,      :QCH4H2O)
    connect_param!(m, :rf_total, :rf_CO₂,     :rf_co2_etminan, :rf_CO₂)

    # Return constructed model.
    return m
end
