
using Mimi
using DataFrames

include("../sneasy/doeclim.jl")
include("../sneasy/ccm.jl")
include("radiativeforcing_hector.jl")
include("../sneasy/rfco2.jl")
include("ohcycle_hector.jl")
include("ch4cycle_hector.jl")
include("o3cycle_hector.jl")
include("rfo3_hector.jl")
include("rfch4h2o_hector.jl")
include("rfch4_hector.jl")
include("../../helper_functions.jl")

function getsneasy_hector(;rcp_scenario="RCP85", start_year = 1765, end_year = 2300)
    m = Model()

    # Calculate number of periods to run model for.
    n_steps = length(start_year:end_year)

    setindex(m, :time, n_steps)

    # TODO: Remove - Set fake indices to read in anomtable parameter (otherwise doesn't work with current MIMI version).
    setindex(m, :anom_row, 100)
    setindex(m, :anom_col, 16000)

    # ---------------------------------------------
    # Create components
    # ---------------------------------------------
    addcomponent(m, ohcyclehector)
    addcomponent(m, ch4cyclehector)
    addcomponent(m, o3cyclehector)
    addcomponent(m, rfo3hector)
    addcomponent(m, rfch4h2ohector)
    addcomponent(m, rfch4hector)
    addcomponent(m, rfco2)
    addcomponent(m, radiativeforcing)
    addcomponent(m, doeclim)
    addcomponent(m, ccm)

    # ---------------------------------------------
    # Read data
    # ---------------------------------------------

    # Select appropirate start year for RCP values (RCPs range from 1765-2500).
    start_index, end_index = findin(collect(1765:2500), [start_year, end_year])

    #----------------------------
    # RCP CONCENTRATION DATA
    #----------------------------
    rcp_conc_raw =readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data", rcp_scenario*"_MIDYEAR_CONCENTRATIONS.csv"), skipstart=37)[start_index:end_index,:]
    rcp_conc = DataFrame(conc_n2o = rcp_conc_raw[:N2O])

    #----------------------------
    # RCP EMISSIONS DATA
    #----------------------------
    rcp_emiss_raw = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)[start_index:end_index,:]
    rcp_emiss = DataFrame(emiss_co2=rcp_emiss_raw[:FossilCO2]+rcp_emiss_raw[:OtherCO2], emiss_ch4=rcp_emiss_raw[:CH4], emiss_nox=rcp_emiss_raw[:NOx], emiss_co=rcp_emiss_raw[:CO], emiss_nmvoc=rcp_emiss_raw[:NMVOC])
    
    #----------------------------
    # RCP RADIATIVE FORCING DATA
    #----------------------------
    rcp_rf_raw = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_MIDYEAR_RADFORCING.csv"), skipstart=58)[start_index:end_index,:]
    # Taking total RCP forcing. Aerosols (direct+indirect) scaled separately. Also subtract modeled RFs: CO₂, CH₄, Trop O₃, Strat H₂O from CH₄
    rcp_forcing = DataFrame(rf_aerosol=(rcp_rf_raw[:TOTAER_DIR_RF]+rcp_rf_raw[:CLOUD_TOT_RF]), rf_other=(rcp_rf_raw[:TOTAL_INCLVOLCANIC_RF]-rcp_rf_raw[:CO2_RF]-rcp_rf_raw[:CH4_RF]-rcp_rf_raw[:TOTAER_DIR_RF]-rcp_rf_raw[:CLOUD_TOT_RF]-rcp_rf_raw[:TROPOZ_RF]-rcp_rf_raw[:CH4OXSTRATH2O_RF]))
      
    #----------------------------
    # OTHER DATA
    #----------------------------
    # Anomaly table for SNEASY
    f_anomtable = readdlm(joinpath(dirname(@__FILE__),"..", "..", "data", "anomtable.txt"))

    rcp_emissions_CO₂ = convert(Array, rcp_emiss[:emiss_co2])
    rcp_emissions_CH₄ = convert(Array, rcp_emiss[:emiss_ch4])
    rcp_emissions_CO = convert(Array, rcp_emiss[:emiss_co]);
    rcp_emissions_NOx = convert(Array, rcp_emiss[:emiss_nox]);
    rcp_emissions_NMVOC = convert(Array, rcp_emiss[:emiss_nmvoc]);
    
    rcp_rf_aerosol = convert(Array, rcp_forcing[:rf_aerosol]);
    rcp_rf_other = convert(Array, rcp_forcing[:rf_other]);
    
    rcp_conc_N₂O = convert(Array, rcp_conc[:conc_n2o])

    anomtable = transpose(f_anomtable)

    # Set user defined F2x_CO₂ value, and calculate scaling term so it agrees with CO2 RF equation.
    # TODO = FIGURE OUT BETTER WAY TO FOLD THIS ALL TOGETHER
    F2x_CO₂ = 3.7
    CO₂_0 = 278.05158
    N₂O_0 = 272.95961
    # Calculate CO₂ radiative forcing scale factor to agree with user-defined forcing from doubling CO₂.
    scale_CO₂ = co2_rf_scale(F2x_CO₂, CO₂_0, N₂O_0)
    CH₄_0 = 721.89411

    # ---------------------------------------------
    # Set parameters
    # ---------------------------------------------
    setparameter(m, :doeclim, :t2co, 2.0)
    setparameter(m, :doeclim, :kappa, 1.1)
    setparameter(m, :doeclim, :T0,     0.0)
    setparameter(m, :doeclim, :H0,     0.0)
    setparameter(m, :doeclim, :F2x_CO₂,  F2x_CO₂)
    setparameter(m, :doeclim, :deltat, 1.0)

    setparameter(m, :ccm, :Q10, 1.311)
    setparameter(m, :ccm, :Beta, 0.502)
    setparameter(m, :ccm, :Eta, 17.7)
    setparameter(m, :ccm, :atmco20, CO₂_0)
    setparameter(m, :ccm, :CO2_emissions, rcp_emissions_CO₂)
    setparameter(m, :ccm, :anomtable, anomtable)
    setparameter(m, :ccm, :deltat, 1.0)
    setparameter(m, :ccm, :oxidised_CH₄_to_CO₂, zeros(n_steps)) #Hector CH4 cycle doesn't have this.

    setparameter(m, :ohcyclehector, :CNOX, 0.0042)
    setparameter(m, :ohcyclehector, :CCO, -0.000105)
    setparameter(m, :ohcyclehector, :CNMVOC, -0.000315)
    setparameter(m, :ohcyclehector, :CCH4, -0.32)
    setparameter(m, :ohcyclehector, :NOX_emissions, rcp_emissions_NOx)
    setparameter(m, :ohcyclehector, :CO_emissions, rcp_emissions_CO)
    setparameter(m, :ohcyclehector, :NMVOC_emissions, rcp_emissions_NMVOC)
    setparameter(m, :ohcyclehector, :TOH0, 6.586) #From Hector output on Github
    setparameter(m, :ohcyclehector, :M0, CH₄_0)

    setparameter(m, :ch4cyclehector, :UC_CH4, 2.78)
    setparameter(m, :ch4cyclehector, :CH4N, 300.)
    setparameter(m, :ch4cyclehector, :Tsoil, 160.)
    setparameter(m, :ch4cyclehector, :Tstrat, 120.)
    setparameter(m, :ch4cyclehector, :M0, CH₄_0)
    setparameter(m, :ch4cyclehector, :CH4_emissions, rcp_emissions_CH₄)

    setparameter(m, :o3cyclehector, :PO3, 32.38) # From Hector output on Github.
    setparameter(m, :o3cyclehector, :NOX_emissions, rcp_emissions_NOx)
    setparameter(m, :o3cyclehector, :CO_emissions, rcp_emissions_CO)
    setparameter(m, :o3cyclehector, :NMVOC_emissions, rcp_emissions_NMVOC)

    setparameter(m, :rfch4h2ohector, :M0, CH₄_0)
    setparameter(m, :rfch4h2ohector, :H₂O_share, 0.05)  

    setparameter(m, :rfch4hector, :N₂O_0, N₂O_0)  #taken from 1850 RCP85 data
    setparameter(m, :rfch4hector, :CH4_0, CH₄_0)
    setparameter(m, :rfch4hector, :a₃, -1.3e-6)
    setparameter(m, :rfch4hector, :b₃, -8.2e-6)
    setparameter(m, :rfch4hector, :N₂O, rcp_conc_N₂O)
    setparameter(m, :rfch4hector, :scale_CH₄, 1.0)

    setparameter(m, :rfco2, :a₁, -2.4e-7)
    setparameter(m, :rfco2, :b₁, 7.2e-4)
    setparameter(m, :rfco2, :c₁, -2.1e-4)
    setparameter(m, :rfco2, :CO₂_0, CO₂_0)
    setparameter(m, :rfco2, :N₂O_0, N₂O_0)  #taken from 1850 RCP85 data
    setparameter(m, :rfco2, :N₂O, rcp_conc_N₂O)
    setparameter(m, :rfco2, :scale_CO₂, scale_CO₂)

    setparameter(m, :radiativeforcing, :rf_aerosol, rcp_rf_aerosol)
    setparameter(m, :radiativeforcing, :rf_other, rcp_rf_other)
    setparameter(m, :radiativeforcing, :alpha, 1.)

    # ---------------------------------------------
    # Connect parameters to variables
    # ---------------------------------------------

    connectparameter(m, :doeclim, :forcing, :radiativeforcing, :rf)
    connectparameter(m, :ccm, :temp, :doeclim, :temp)
    connectparameter(m, :rfco2, :CO₂, :ccm, :atmco2)
    connectparameter(m, :ohcyclehector, :CH4, :ch4cyclehector, :CH4)
    connectparameter(m, :ch4cyclehector, :TOH, :ohcyclehector, :TOH)
    connectparameter(m, :o3cyclehector, :CH4, :ch4cyclehector, :CH4)
    connectparameter(m, :rfo3hector, :O3, :o3cyclehector, :O3)
    connectparameter(m, :rfch4h2ohector, :CH4, :ch4cyclehector, :CH4)
    connectparameter(m, :rfch4hector, :CH4, :ch4cyclehector, :CH4)
    connectparameter(m, :radiativeforcing, :rf_CH4, :rfch4hector, :rf_CH4)
    connectparameter(m, :radiativeforcing, :rf_O3, :rfo3hector, :rf_O3)
    connectparameter(m, :radiativeforcing, :rf_ch4h2o, :rfch4h2ohector, :rf_ch4h2o)
    connectparameter(m, :radiativeforcing, :rf_co2, :rfco2, :rf_co2)

    return m
end
