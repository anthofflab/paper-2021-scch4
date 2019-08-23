
using Mimi
using DataFrames

include("../sneasy/doeclim.jl")
include("../sneasy/ccm.jl")
include("radiativeforcing_fair.jl")
include("../sneasy/rfco2.jl")
#include("ch4cycle_magicc.jl")
include("ch4cycle_fair.jl")
include("rfch4_fair.jl")
include("rfo3_fair.jl")
include("../../helper_functions.jl")

function getsneasy_fair(;rcp_scenario="RCP85", start_year = 1765, end_year = 2300)
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

    addcomponent(m, ch4cycle_fair)
    addcomponent(m, rfco2)
    addcomponent(m, rfch4fair)
    addcomponent(m, fair_trop_o3)
    addcomponent(m, radiativeforcing)
    addcomponent(m, doeclim)
    addcomponent(m, ccm)

    # ---------------------------------------------
    # Read data
    # ---------------------------------------------

    #Read in RF data for CH4 separately because it is not included in forcing_rcp85.txt.
    #rcp_data_conc =readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data", rcp_scenario*"_MIDYEAR_CONCENTRATIONS.csv"), skipstart=37);

    #Foss_Hist = readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data","FOSS_HIST_modified_timestep.csv"), separator = ',', header=true);
    #rf_ch4_data = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_MIDYEAR_RADFORCING.csv"), skipstart=58);
    #rf_ch4_data = DataFrame(year=rf_ch4_data[:YEARS], CH4_RF=rf_ch4_data[:CH4_RF], ch4_h20_rf = rf_ch4_data[:CH4OXSTRATH2O_RF], TROPOZ_RF = rf_ch4_data[:TROPOZ_RF])

    #f_anomtable = readdlm(joinpath(dirname(@__FILE__),"..", "..", "data", "anomtable.txt"))
    #rf_data = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", "forcing_rcp85.txt"), separator = ' ', header=true)
    #df = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)
    #rename!(df, :YEARS, :year);
    #df = join(df, rf_data, on=:year, kind=:outer)
    #df = join(df, rf_ch4_data, on=:year, kind=:outer)

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
    # Time-varying natural CH₄ Emissions from FAIR 1.3
    fair_natural_raw = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", "FAIR_natural_emissions.csv"), skipstart=3);
    # Time-varying, RCP-specific "fossil CH4 fraction as a proportion of total anthropogenic CH4 emissions" from FAIR 1.3
    fossil_CH₄_raw = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", "FAIR_fossilCH4_fraction.csv"), skipstart=4);

    #rf_other=df[:ghg_nonco2]+df[:volcanic]+df[:solar]+df[:other]- df[:CH4_RF] - df[:ch4_h20_rf] - df[:TROPOZ_RF])
    #Subtract strat h20 from ch4 oxidation, tropospheric ozone, and ch4 radiative forcing
    #df = DataFrame(year=df[:year], co2=df[:FossilCO2]+df[:OtherCO2], ch4=df[:CH4], nox=df[:NOx], co=df[:CO], nmvoc=df[:NMVOC], rf_aerosol=df[:aerosol_direct]+df[:aerosol_indirect], rf_other=df[:ghg_nonco2]+df[:volcanic]+df[:solar]+df[:other]- df[:CH4_RF] - df[:ch4_h20_rf] - df[:TROPOZ_RF])

    #df = @where(df, :year .>= start_year)
    #df = @where(df, :year .<= end_year)

    rcp_emissions_CO₂ = convert(Array, rcp_emiss[:emiss_co2])
    rcp_emissions_CH₄ = convert(Array, rcp_emiss[:emiss_ch4])
    rcp_emissions_CO = convert(Array, rcp_emiss[:emiss_co]);
    rcp_emissions_NOx = convert(Array, rcp_emiss[:emiss_nox]);
    rcp_emissions_NMVOC = convert(Array, rcp_emiss[:emiss_nmvoc]);
    
    emissions_natural_CH₄ = convert(Array,fair_natural_raw[:ch4])

    rcp_rf_aerosol = convert(Array, rcp_forcing[:rf_aerosol]);
    rcp_rf_other = convert(Array, rcp_forcing[:rf_other]);
    
    rcp_conc_N₂O = convert(Array, rcp_conc[:conc_n2o])

    fossil_CH₄_fraction = convert(Array, fossil_CH₄_raw[Symbol(rcp_scenario)])

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

    setparameter(m, :ch4cycle_fair, :fossil_emiss, rcp_emissions_CH₄)
    setparameter(m, :ch4cycle_fair, :natural_emiss, emissions_natural_CH₄)
    setparameter(m, :ch4cycle_fair, :CH₄_0, CH₄_0)
    setparameter(m, :ch4cycle_fair, :τ, 9.3)
    setparameter(m, :ch4cycle_fair, :fossil_frac, fossil_CH₄_fraction)
    setparameter(m, :ch4cycle_fair, :oxidation_frac, 0.61)
    setparameter(m, :ch4cycle_fair, :constant_natural_CH₄, 190.5807)
    setparameter(m, :ch4cycle_fair, :model_start_year, start_year)

    setparameter(m, :rfch4fair, :N₂O_0, N₂O_0)  #taken from 1765 RCP85 data
    setparameter(m, :rfch4fair, :H₂O_share, 0.12)  
    setparameter(m, :rfch4fair, :scale_CH₄, 1.0)
    setparameter(m, :rfch4fair, :CH₄_0, CH₄_0)
    setparameter(m, :rfch4fair, :a₃, -1.3e-6)
    setparameter(m, :rfch4fair, :b₃, -8.2e-6)
    setparameter(m, :rfch4fair, :N₂O, rcp_conc_N₂O)

    setparameter(m, :fair_trop_o3, :model_start_year, start_year)
    setparameter(m, :fair_trop_o3, :NOx_emissions, rcp_emissions_NOx)
    setparameter(m, :fair_trop_o3, :CO_emissions, rcp_emissions_CO)
    setparameter(m, :fair_trop_o3, :NMVOC_emissions, rcp_emissions_NMVOC)
    setparameter(m, :fair_trop_o3, :CH₄_0, CH₄_0)
    setparameter(m, :fair_trop_o3, :T0, 0.0)

    setparameter(m, :rfco2, :a₁, -2.4e-7)
    setparameter(m, :rfco2, :b₁, 7.2e-4)
    setparameter(m, :rfco2, :c₁, -2.1e-4)
    setparameter(m, :rfco2, :CO₂_0, CO₂_0)
    setparameter(m, :rfco2, :scale_CO₂, scale_CO₂)
    setparameter(m, :rfco2, :N₂O_0, N₂O_0)  #taken from 1765 RCP85 data
    setparameter(m, :rfco2, :N₂O, rcp_conc_N₂O)

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

    connectparameter(m, :rfco2, :CO₂, :ccm, :atmco2)
    connectparameter(m, :rfch4fair, :CH₄, :ch4cycle_fair, :CH₄)
    connectparameter(m, :fair_trop_o3, :temperature, :doeclim, :temp)
    connectparameter(m, :fair_trop_o3, :CH₄, :ch4cycle_fair, :CH₄)
    connectparameter(m, :radiativeforcing, :rf_co2, :rfco2, :rf_co2)
    connectparameter(m, :radiativeforcing, :forcing_CH₄_H₂O, :rfch4fair, :forcing_CH₄_H₂O)
    connectparameter(m, :radiativeforcing, :forcing_CH₄, :rfch4fair, :forcing_CH₄)
    connectparameter(m, :radiativeforcing, :forcing_O₃, :fair_trop_o3, :forcing_O₃)
    connectparameter(m, :doeclim, :forcing, :radiativeforcing, :rf)
    connectparameter(m, :ccm, :temp, :doeclim, :temp)
    connectparameter(m, :ccm, :oxidised_CH₄_to_CO₂, :ch4cycle_fair, :oxidised_CH₄_to_CO₂)

    return m
end
