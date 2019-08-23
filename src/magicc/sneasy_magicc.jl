
using Mimi
using DataFrames

include("../sneasy/doeclim.jl")
include("../sneasy/ccm.jl")
include("radiativeforcing_magicc.jl")
include("../sneasy/rfco2.jl")
#include("ch4cycle_magicc.jl")
include("ch4cycle_magicc_mcmc_emissions_version.jl")
include("rfch4_magicc.jl")
include("rfo3_magicc.jl")
include("../../helper_functions.jl")

function getsneasy_magicc(;rcp_scenario="RCP85", start_year = 1765, end_year = 2300)
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

    addcomponent(m, ch4cyclemagicc)
    addcomponent(m, rfco2)
    addcomponent(m, rfch4magicc)
    addcomponent(m, rfo3magicc)
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
    # Foss history time series from MAGICC
    Foss_Hist = readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data","FOSS_HIST_modified_timestep.csv"), separator = ',', header=true);

    rcp_emissions_CO₂ = convert(Array, rcp_emiss[:emiss_co2])
    rcp_emissions_CH₄ = convert(Array, rcp_emiss[:emiss_ch4])
    rcp_emissions_CO = convert(Array, rcp_emiss[:emiss_co]);
    rcp_emissions_NOx = convert(Array, rcp_emiss[:emiss_nox]);
    rcp_emissions_NMVOC = convert(Array, rcp_emiss[:emiss_nmvoc]);
    
    rcp_rf_aerosol = convert(Array, rcp_forcing[:rf_aerosol]);
    rcp_rf_other = convert(Array, rcp_forcing[:rf_other]);
    
    rcp_conc_N₂O = convert(Array, rcp_conc[:conc_n2o])

    f_foss_hist = convert(Array, Foss_Hist[:EFOSS])

    # Timesteps
    deltat = 1.0
    #Pre-industrial CH4 concentrations

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

    setparameter(m, :ch4cyclemagicc, :deltat, deltat)
    setparameter(m, :ch4cyclemagicc, :BBCH4, 2.78)
    setparameter(m, :ch4cyclemagicc, :ANOX, 0.0042)
    setparameter(m, :ch4cyclemagicc, :ACO, -0.000105)
    setparameter(m, :ch4cyclemagicc, :AVOC, -0.000315)
    setparameter(m, :ch4cyclemagicc, :TAUINIT, 7.73) # Lifetime that produces close match to concnetration version of MAGICC.
    setparameter(m, :ch4cyclemagicc, :SCH4, -0.32)
    setparameter(m, :ch4cyclemagicc, :TAUSOIL, 160.)
    setparameter(m, :ch4cyclemagicc, :TAUSTRAT, 120.)
    setparameter(m, :ch4cyclemagicc, :GAM, -1.0)
    setparameter(m, :ch4cyclemagicc, :fffrac, 0.18)
    setparameter(m, :ch4cyclemagicc, :CH4_emissions, rcp_emissions_CH₄)
    setparameter(m, :ch4cyclemagicc, :NOX_emissions, rcp_emissions_NOx)
    setparameter(m, :ch4cyclemagicc, :CO_emissions, rcp_emissions_CO)
    setparameter(m, :ch4cyclemagicc, :NMVOC_emissions, rcp_emissions_NMVOC)
    setparameter(m, :ch4cyclemagicc, :CH4_natural, 266.5)
    setparameter(m, :ch4cyclemagicc, :CH4_0, CH₄_0)  #Pre-industrial CH4 concentrations (value for 1765 from RCP8.5 concnetration data)

    setparameter(m, :rfch4magicc, :N₂O_0, N₂O_0)  #taken from 1765 RCP85 data
    setparameter(m, :rfch4magicc, :STRATH2O, 0.15)  #Documentation says 0.15 not 0.05 from GCAM
    setparameter(m, :rfch4magicc, :TROZSENS, 0.042)
    setparameter(m, :rfch4magicc, :OZCH4, 5.0)
    setparameter(m, :rfch4magicc, :CH4_0, CH₄_0)
    setparameter(m, :rfch4magicc, :a₃, -1.3e-6)
    setparameter(m, :rfch4magicc, :b₃, -8.2e-6)
    setparameter(m, :rfch4magicc, :N₂O, rcp_conc_N₂O)
    setparameter(m, :rfch4magicc, :scale_CH₄, 1.0)

    setparameter(m, :rfo3magicc, :deltat, deltat)
    setparameter(m, :rfo3magicc, :NOX_emissions, rcp_emissions_NOx)
    setparameter(m, :rfo3magicc, :CO_emissions, rcp_emissions_CO)
    setparameter(m, :rfo3magicc, :NMVOC_emissions, rcp_emissions_NMVOC)
    setparameter(m, :rfo3magicc, :OZNOX, 0.125)
    setparameter(m, :rfo3magicc, :OZCO, 0.0011)
    setparameter(m, :rfo3magicc, :OZVOC, 0.0033)
    setparameter(m, :rfo3magicc, :OZ00CH4, 0.161)
    setparameter(m, :rfo3magicc, :FOSSHIST, f_foss_hist)
    setparameter(m, :rfo3magicc, :TROZSENS, 0.042)

    setparameter(m, :rfco2, :a₁, -2.4e-7)
    setparameter(m, :rfco2, :b₁, 7.2e-4)
    setparameter(m, :rfco2, :c₁, -2.1e-4)
    setparameter(m, :rfco2, :CO₂_0, 278.05158)
    setparameter(m, :rfco2, :N₂O_0, N₂O_0)  #taken from 1765 RCP85 data
    setparameter(m, :rfco2, :N₂O, rcp_conc_N₂O)
    setparameter(m, :rfco2, :scale_CO₂, scale_CO₂)

    setparameter(m, :radiativeforcing, :deltat, deltat)
    setparameter(m, :radiativeforcing, :rf_aerosol, rcp_rf_aerosol)
    setparameter(m, :radiativeforcing, :rf_other, rcp_rf_other)
    setparameter(m, :radiativeforcing, :alpha, 1.)

    setparameter(m, :doeclim, :t2co, 2.0)
    setparameter(m, :doeclim, :kappa, 1.1)
    setparameter(m, :doeclim, :deltat, deltat)
    setparameter(m, :doeclim, :T0,     0.0)
    setparameter(m, :doeclim, :H0,     0.0)
    setparameter(m, :doeclim, :F2x_CO₂,  F2x_CO₂)

    setparameter(m, :ccm, :deltat, deltat)
    setparameter(m, :ccm, :Q10, 1.311)
    setparameter(m, :ccm, :Beta, 0.502)
    setparameter(m, :ccm, :Eta, 17.7)
    setparameter(m, :ccm, :atmco20, CO₂_0)
    setparameter(m, :ccm, :CO2_emissions, rcp_emissions_CO₂)
    setparameter(m, :ccm, :anomtable, anomtable)

    # ---------------------------------------------
    # Connect parameters to variables
    # ---------------------------------------------

    connectparameter(m, :ch4cyclemagicc, :temp, :doeclim, :temp)
    connectparameter(m, :rfco2, :CO₂, :ccm, :atmco2)
    connectparameter(m, :rfch4magicc, :CH4, :ch4cyclemagicc, :CH4)
    connectparameter(m, :rfo3magicc, :QCH4OZ, :rfch4magicc, :QCH4OZ)
    connectparameter(m, :radiativeforcing, :rf_co2, :rfco2, :rf_co2)
    connectparameter(m, :radiativeforcing, :QCH4H2O, :rfch4magicc, :QCH4H2O)
    connectparameter(m, :radiativeforcing, :QMeth, :rfch4magicc, :QMeth)
    connectparameter(m, :radiativeforcing, :rf_O3, :rfo3magicc, :rf_O3)
    connectparameter(m, :doeclim, :forcing, :radiativeforcing, :rf)
    connectparameter(m, :ccm, :temp, :doeclim, :temp)
    connectparameter(m, :ccm, :oxidised_CH₄_to_CO₂, :ch4cyclemagicc, :emeth)

    return m
end
