#-------------------------------------------------------------------------------
# This function creates an instance of SNEASY+Hector.
#-------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiHECTOR
using MimiSNEASY
using DataFrames
using CSVFiles

# Load required files.
include(joinpath("additional_model_components", "rf_total.jl"))
include(joinpath("additional_model_components", "rf_co2.jl"))
include(joinpath("additional_model_components", "rf_ch4.jl"))
include(joinpath("..", "helper_functions.jl"))

function create_sneasy_hectorch4(;rcp_scenario::String="RCP85", start_year::Int=1765, end_year::Int=2300, etminan_ch4_forcing::Bool=true)

    # ---------------------------------------------
    # Load and clean up necessary data.
    # ---------------------------------------------

 	# Load RCP emissions and concentration scenario values (RCP options = "RCP26" and "RCP85").
    rcp_emissions      = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_emissions.csv"), skiplines_begin=36))
    rcp_concentrations = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_concentrations.csv"), skiplines_begin=37))
    rcp_forcing 	   = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_midyear_radforcings.csv"), skiplines_begin=58))

  	# Find start and end year indices to crop RCP scenario data to correct model time horizon.
    rcp_indices = findall((in)(collect(start_year:end_year)), rcp_emissions.YEARS)

    # Set pre-industrial atmospheric CO₂, CH₄, and N₂O concentrations to RCP values in 1765.
    CO₂_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CO2][1]
    CH₄_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :CH4][1]
    N₂O_0 = rcp_concentrations[rcp_concentrations.YEARS .== 1765, :N2O][1]

    # Calculate aerosol and exogenous RCP radiative forcing scenarios.
    rcp_aerosol_forcing   = rcp_forcing.TOTAER_DIR_RF .+ rcp_forcing.CLOUD_TOT_RF
    rcp_exogenous_forcing = rcp_forcing.TOTAL_INCLVOLCANIC_RF .- rcp_forcing.CO2_RF .- rcp_forcing.CH4_RF .- rcp_forcing.TROPOZ_RF .- rcp_forcing.CH4OXSTRATH2O_RF .- rcp_aerosol_forcing


 	# ------------------------------------------------------------
    # Initialize Mimi-SNEASY and add new CH₄ components to model.
    # ------------------------------------------------------------

    # Get an instance of Mimi-SNEASY.
	m = MimiSNEASY.getsneasy(start_year=start_year, end_year=end_year)

	# Remove old radiative forcing components.
    delete!(m, :rfco2)
    delete!(m, :radiativeforcing)

    # Add in new components.
    add_comp!(m, rf_total, before = :doeclim)
    add_comp!(m, rf_co2_etminan, before = :rf_total)
    add_comp!(m, MimiHECTOR.rf_ch4h2o, before = :rf_co2_etminan)
    add_comp!(m, MimiHECTOR.rf_o3, before = :rf_ch4h2o)
    add_comp!(m, MimiHECTOR.ch4_cycle, before = :rf_o3)
    add_comp!(m, MimiHECTOR.oh_cycle, before = :ch4_cycle)

    # Add in user-specified CH₄ radiative forcing component.
    # Note: If not using Etminan et al. equations, use original forcing equations from parent CH₄ model.
    if etminan_ch4_forcing == true
    	add_comp!(m, rf_ch4_etminan, before = :rf_ch4h2o)
    else
    	add_comp!(m, MimiHECTOR.rf_ch4, before = :rf_ch4h2o)
    end


    # ---------------------------------------------
    # Set component parameters.
    # ---------------------------------------------

 	# ---- Tropospheric Sink (OH) Lifetime ---- #
    set_param!(m, :oh_cycle, :CNOX, 0.0042)
    set_param!(m, :oh_cycle, :CCO, -0.000105)
    set_param!(m, :oh_cycle, :CNMVOC, -0.000315)
    set_param!(m, :oh_cycle, :CCH4, -0.32)
    set_param!(m, :oh_cycle, :NOX_emissions, rcp_emissions.NOx[rcp_indices])
    set_param!(m, :oh_cycle, :CO_emissions, rcp_emissions.CO[rcp_indices])
    set_param!(m, :oh_cycle, :NMVOC_emissions, rcp_emissions.NMVOC[rcp_indices])
    set_param!(m, :oh_cycle, :TOH0, 6.586)
    set_param!(m, :oh_cycle, :M0, CH₄_0)

    # ---- Methane Cycle ---- #
    set_param!(m, :ch4_cycle, :UC_CH4, 2.78)
    set_param!(m, :ch4_cycle, :CH4N, 300.)
    set_param!(m, :ch4_cycle, :Tsoil, 160.)
    set_param!(m, :ch4_cycle, :Tstrat, 120.)
    set_param!(m, :ch4_cycle, :M0, CH₄_0)
    set_param!(m, :ch4_cycle, :CH4_emissions, rcp_emissions.CH4[rcp_indices])

    # ---- Methane Radiative Forcing ---- #
    if etminan_ch4_forcing == true
        set_param!(m, :rf_ch4_etminan, :CH₄_0, CH₄_0)
        set_param!(m, :rf_ch4_etminan, :N₂O_0, N₂O_0)
        set_param!(m, :rf_ch4_etminan, :scale_CH₄, 1.0)
        set_param!(m, :rf_ch4_etminan, :a₃, -1.3e-6)
        set_param!(m, :rf_ch4_etminan, :b₃, -8.2e-6)
        set_param!(m, :rf_ch4_etminan, :N₂O, rcp_concentrations.N2O[rcp_indices])
    else
        set_param!(m, :rf_ch4, :N₂O_0, N₂O_0)
        set_param!(m, :rf_ch4, :CH4_0, CH₄_0)
        set_param!(m, :rf_ch4, :scale_CH₄, 1.0)
    end

    # ---- Straospheric Water Vapor From Methane Radiative Forcing ---- #
    set_param!(m, :rf_ch4h2o, :M0, CH₄_0)
    set_param!(m, :rf_ch4h2o, :H₂O_share, 0.05)

    # ---- Tropospheric Ozone Radiative Forcing ---- #
    set_param!(m, :rf_o3, :O₃_0, 32.38)
    set_param!(m, :rf_o3, :NOx_emissions, rcp_emissions.NOx[rcp_indices])
    set_param!(m, :rf_o3, :CO_emissions, rcp_emissions.CO[rcp_indices])
    set_param!(m, :rf_o3, :NMVOC_emissions, rcp_emissions.NMVOC[rcp_indices])

    # ---- Carbon Dioxide Radiative Forcing ---- #
    set_param!(m, :rf_co2_etminan, :a₁, -2.4e-7)
    set_param!(m, :rf_co2_etminan, :b₁, 7.2e-4)
    set_param!(m, :rf_co2_etminan, :c₁, -2.1e-4)
    set_param!(m, :rf_co2_etminan, :CO₂_0, CO₂_0)
    set_param!(m, :rf_co2_etminan, :N₂O_0, N₂O_0)
    set_param!(m, :rf_co2_etminan, :N₂O, rcp_concentrations.N2O[rcp_indices])
    set_param!(m, :rf_co2_etminan, :rf_scale_CO₂, co2_rf_scale(3.7, CO₂_0, N₂O_0))

    # ---- Total Radiative Forcing ---- #
    set_param!(m, :rf_total, :α, 1.0)
  	set_param!(m, :rf_total, :rf_aerosol, rcp_aerosol_forcing[rcp_indices])
    set_param!(m, :rf_total, :rf_exogenous, rcp_exogenous_forcing[rcp_indices])


	# ----------------------------------------------------------
    # Create connections between Mimi SNEASY+Hector components.
    # ----------------------------------------------------------
    connect_param!(m, :doeclim,        :forcing, :rf_total,  :total_forcing)
    connect_param!(m, :ccm,            :temp,    :doeclim,   :temp)
    connect_param!(m, :rf_co2_etminan, :CO₂,     :ccm,       :atmco2)
    connect_param!(m, :oh_cycle,       :CH4,     :ch4_cycle, :CH4)
    connect_param!(m, :ch4_cycle,      :TOH,     :oh_cycle,  :TOH)
    connect_param!(m, :rf_o3,          :CH₄,     :ch4_cycle, :CH4)
    connect_param!(m, :rf_ch4h2o,      :CH4,     :ch4_cycle, :CH4)

    # Create different connections if using updated Etminan et al. CH₄ forcing equations.
    if etminan_ch4_forcing == true
  	    connect_param!(m, :rf_ch4_etminan, :CH₄,    :ch4_cycle,      :CH4)
    	connect_param!(m, :rf_total,       :rf_CH₄, :rf_ch4_etminan, :rf_CH₄)
    else
  	    connect_param!(m, :rf_ch4,   :CH4,    :ch4_cycle, :CH4)
    	connect_param!(m, :rf_total, :rf_CH₄, :rf_ch4,    :rf_CH4)
    end

    connect_param!(m, :rf_total, :rf_O₃,      :rf_o3,          :rf_O₃)
    connect_param!(m, :rf_total, :rf_CH₄_H₂O, :rf_ch4h2o,      :rf_ch4h2o)
    connect_param!(m, :rf_total, :rf_CO₂,     :rf_co2_etminan, :rf_CO₂)

    # Return constructed model.
    return m
end
