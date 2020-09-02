#-------------------------------------------------------------------------------
# This function creates an instance of SNEASY+FAIR.
#-------------------------------------------------------------------------------

# Load required packages.
using Mimi
using MimiFAIR
using MimiSNEASY
using DataFrames
using CSVFiles

# Load required files.
include(joinpath("additional_model_components", "rf_total.jl"))
include(joinpath("additional_model_components", "rf_co2_etminan.jl"))
include(joinpath("additional_model_components", "rf_ch4_myhre_fair.jl"))
include(joinpath("additional_model_components", "total_co2_emissions.jl"))
include(joinpath("..", "helper_functions.jl"))

function create_sneasy_fairch4(;rcp_scenario::String="RCP85", start_year::Int=1765, end_year::Int=2300, etminan_ch4_forcing::Bool=true)

    # ---------------------------------------------
    # Load and clean up necessary data.
    # ---------------------------------------------

    # Load RCP emissions and concentration scenario values (RCP options = "RCP26" and "RCP85").
    rcp_emissions      = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_emissions.csv"), skiplines_begin=36))
    rcp_concentrations = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_concentrations.csv"), skiplines_begin=37))
    rcp_forcing        = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", rcp_scenario*"_midyear_radforcings.csv"), skiplines_begin=58))

    # Load RCP scenario data for fossil CH₄ fraction as a proportion of total anthropogenic CH₄ emissions.
    rcp_fossil_ch4_frac = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", "FAIR_Fossil_CH4_fraction.csv"), skiplines_begin=4))[:,Symbol(rcp_scenario)]

    # Load FAIR greenhouse species data.
    fair_ghg_data = DataFrame(load(joinpath(@__DIR__, "..", "..", "data", "model_data", "FAIR_GHG_species_data.csv"), skiplines_begin=10))

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

    # Calculate CH₄ emission to concentration conversion (based on atmospheric mas of 5.1352e18 kg).
    mol_wt_air = fair_ghg_data[fair_ghg_data.gas .== "AIR", :mol_weight][1]
    mol_wt_ch4 = fair_ghg_data[fair_ghg_data.gas .== "CH4", :mol_weight][1]
    CH₄_emiss2conc = (5.1352e18/1.0e18) * (mol_wt_ch4 / mol_wt_air)

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
    add_comp!(m, MimiFAIR.trop_o3_rf, before = :rf_co2_etminan)
    add_comp!(m, total_co2_emissions, before= :trop_o3_rf)
    add_comp!(m, MimiFAIR.ch4_cycle, before = :total_co2_emissions)

    # Add in user-specified CH₄ radiative forcing component.
    # Note: FAIR uses Etminan et al. CH₄ radiative forcing equations. If not using these, switch to older Myhre et al. equations.
    if etminan_ch4_forcing == true
        add_comp!(m, MimiFAIR.ch4_rf, before = :rf_co2_etminan)
    else
        add_comp!(m, rf_ch4_myhre_fair, before = :rf_co2_etminan)
    end


    # ---------------------------------------------
    # Set component parameters.
    # ---------------------------------------------

    # ---- Carbon Cycle ---- #
    update_param!(m, :atmco20, CO₂_0)

    # ---- Methane Cycle ---- #
    set_param!(m, :ch4_cycle, :fossil_emiss_CH₄, rcp_emissions.CH4[rcp_indices])
    set_param!(m, :ch4_cycle, :natural_emiss_CH₄, ones(length(start_year:end_year)).*190.5807)    
    set_param!(m, :ch4_cycle, :τ_CH₄, 9.3)
    set_param!(m, :ch4_cycle, :fossil_frac, rcp_fossil_ch4_frac[rcp_indices])
    set_param!(m, :ch4_cycle, :oxidation_frac, 0.61)
    set_param!(m, :ch4_cycle, :mol_weight_CH₄, fair_ghg_data[fair_ghg_data.gas .== "CH4", :mol_weight][1])
    set_param!(m, :ch4_cycle, :mol_weight_CO₂, fair_ghg_data[fair_ghg_data.gas .== "CO2", :mol_weight][1])
    set_param!(m, :ch4_cycle, :emiss2conc_ch4, CH₄_emiss2conc)

    # ---- Total Carbon Dioxide Emissions ---- #
    set_param!(m, :total_co2_emissions, :exogenous_CO₂_emissions, rcp_co2_emissions[rcp_indices])

    # ---- Methane Radiative Forcing ---- #
    if etminan_ch4_forcing == true
        set_param!(m, :ch4_rf, :scale_CH₄, 1.0)
        set_param!(m, :ch4_rf, :a₃, -1.3e-6)
        set_param!(m, :ch4_rf, :b₃, -8.2e-6)
        set_param!(m, :ch4_rf, :H₂O_share, 0.12)
    else
        set_param!(m, :rf_ch4_myhre_fair, :scale_CH₄, 1.0)
        set_param!(m, :rf_ch4_myhre_fair, :H₂O_share, 0.15)
    end

    # ---- Tropospheric Ozone Radiative Forcing ---- #
    set_param!(m, :trop_o3_rf, :NOx_emissions, rcp_emissions.NOx[rcp_indices])
    set_param!(m, :trop_o3_rf, :CO_emissions, rcp_emissions.CO[rcp_indices])
    set_param!(m, :trop_o3_rf, :NMVOC_emissions, rcp_emissions.NMVOC[rcp_indices])
    set_param!(m, :trop_o3_rf, :mol_weight_N, fair_ghg_data[fair_ghg_data.gas .== "N", :mol_weight][1])
    set_param!(m, :trop_o3_rf, :mol_weight_NO, fair_ghg_data[fair_ghg_data.gas .== "NO", :mol_weight][1])
    set_param!(m, :trop_o3_rf, :T0, 0.0)
    set_param!(m, :trop_o3_rf, :fix_pre1850_RCP, true)

    # ---- Carbon Dioxide Radiative Forcing ---- #
    set_param!(m, :rf_co2_etminan, :a₁, -2.4e-7)
    set_param!(m, :rf_co2_etminan, :b₁, 7.2e-4)
    set_param!(m, :rf_co2_etminan, :c₁, -2.1e-4)
    set_param!(m, :rf_co2_etminan, :CO₂_0, CO₂_0)
    set_param!(m, :rf_co2_etminan, :rf_scale_CO₂, co2_rf_scale(3.7, CO₂_0, N₂O_0))

    # ---- Total Radiative Forcing ---- #
    set_param!(m, :rf_total, :α, 1.0)
    connect_param!(m, :rf_total, :rf_aerosol, :rf_aerosol)
    update_param!(m, :rf_aerosol, rcp_aerosol_forcing[rcp_indices])
    set_param!(m, :rf_total, :rf_exogenous, rcp_exogenous_forcing[rcp_indices])

    # ---- Common parameters ----
    set_param!(m, :CH₄_0, CH₄_0)
    set_param!(m, :N₂O_0, N₂O_0)
    set_param!(m, :N₂O, rcp_concentrations.N2O[rcp_indices])

    # ----------------------------------------------------------
    # Create connections between Mimi SNEASY+Hector components.
    # ----------------------------------------------------------
    connect_param!(m, :doeclim,             :forcing,             :rf_total,            :total_forcing)
    connect_param!(m, :total_co2_emissions, :oxidized_CH₄_to_CO₂, :ch4_cycle,           :oxidised_CH₄)
    connect_param!(m, :ccm,                 :CO2_emissions,       :total_co2_emissions, :total_CO₂_emissions)
    connect_param!(m, :ccm,                 :temp,                :doeclim,             :temp)
    connect_param!(m, :rf_co2_etminan,      :CO₂,                 :ccm,                 :atmco2)
    connect_param!(m, :trop_o3_rf,          :CH₄,                 :ch4_cycle,           :CH₄)
    connect_param!(m, :trop_o3_rf,          :temperature,         :doeclim,             :temp)

    # Create different connections if using updated Etminan et al. CH₄ forcing equations.
    if etminan_ch4_forcing == true
        connect_param!(m, :ch4_rf,            :CH₄,        :ch4_cycle,         :CH₄)
        connect_param!(m, :rf_total,          :rf_CH₄,     :ch4_rf,            :forcing_CH₄)
        connect_param!(m, :rf_total,          :rf_CH₄_H₂O, :ch4_rf,            :forcing_CH₄_H₂O)

    else
        connect_param!(m, :rf_ch4_myhre_fair, :CH₄,        :ch4_cycle,         :CH₄)
        connect_param!(m, :rf_total,          :rf_CH₄,     :rf_ch4_myhre_fair, :forcing_CH₄)
        connect_param!(m, :rf_total,          :rf_CH₄_H₂O, :rf_ch4_myhre_fair, :forcing_CH₄_H₂O)

    end

    connect_param!(m, :rf_total, :rf_O₃,  :trop_o3_rf,     :forcing_trop_O₃)
    connect_param!(m, :rf_total, :rf_CO₂, :rf_co2_etminan, :rf_CO₂)

    # Return constructed model.
    return m
end
