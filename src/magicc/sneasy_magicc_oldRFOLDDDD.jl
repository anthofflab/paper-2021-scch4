
using Mimi
using DataFrames

include("doeclim.jl")
include("ccm.jl")
include("radiativeforcing_magicc.jl")
include("rfco2.jl")
#include("ch4cycle_magicc.jl")
include("ch4cycle_magicc_mcmc_emissions_version.jl")
include("rfch4_magicc_old.jl")
include("rfo3_magicc.jl")


function getsneasy_oldRF(;rcp_scenario="RCP85", start_year = 1850, end_year = 2300)
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

    addcomponent(m, ch4cyclemagicccomponent.ch4cyclemagicc, :ch4cyclemagicc)
    addcomponent(m, rfco2component.rfco2, :rfco2)
    addcomponent(m, rfch4magicccomponent.rfch4magicc, :rfch4magicc)
    addcomponent(m, rfo3magicccomponent.rfo3magicc, :rfo3magicc)
    addcomponent(m, radiativeforcingcomponent.radiativeforcing, :radiativeforcing)
    addcomponent(m, doeclimcomponent.doeclim, :doeclim)
    addcomponent(m, ccmcomponent.ccm, :ccm)

    # ---------------------------------------------
    # Read data
    # ---------------------------------------------

    #Read in RF data for CH4 separately because it is not included in forcing_rcp85.txt.
    rcp_data_conc =readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data", rcp_scenario*"_MIDYEAR_CONCENTRATIONS.csv"), skipstart=37);

    Foss_Hist = readtable(joinpath(dirname(@__FILE__),"..","..","calibration","data","FOSS_HIST_modified_timestep.csv"), separator = ',', header=true);
    rf_ch4_data = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_MIDYEAR_RADFORCING.csv"), skipstart=58);
    rf_ch4_data = DataFrame(year=rf_ch4_data[:YEARS], CH4_RF=rf_ch4_data[:CH4_RF], ch4_h20_rf = rf_ch4_data[:CH4OXSTRATH2O_RF], TROPOZ_RF = rf_ch4_data[:TROPOZ_RF])

    f_anomtable = readdlm(joinpath(dirname(@__FILE__),"..", "..", "data", "anomtable.txt"))
    rf_data = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", "forcing_rcp85.txt"), separator = ' ', header=true)
    df = readtable(joinpath(dirname(@__FILE__),"..", "..", "calibration", "data", rcp_scenario*"_EMISSIONS.csv"), skipstart=36)
    rename!(df, :YEARS, :year);
    df = join(df, rf_data, on=:year, kind=:outer)
    df = join(df, rf_ch4_data, on=:year, kind=:outer)

    #Subtract strat h20 from ch4 oxidation, tropospheric ozone, and ch4 radiative forcing
    df = DataFrame(year=df[:year], co2=df[:FossilCO2]+df[:OtherCO2], ch4=df[:CH4], nox=df[:NOx], co=df[:CO], nmvoc=df[:NMVOC], rf_aerosol=df[:aerosol_direct]+df[:aerosol_indirect], rf_other=df[:ghg_nonco2]+df[:volcanic]+df[:solar]+df[:other]- df[:CH4_RF] - df[:ch4_h20_rf] - df[:TROPOZ_RF])

    # Select appropirate start year for forcing values.
    df = @where(df, :year .>= start_year)
    df = @where(df, :year .<= end_year)

    f_co2emissions = convert(Array, df[:co2])
    f_ch4emissions = convert(Array, df[:ch4])
    f_coemissions = convert(Array, df[:co]);
    f_noxemissions = convert(Array, df[:nox]);
    f_nmvocemissions = convert(Array, df[:nmvoc]);
    f_rfaerosol = convert(Array, df[:rf_aerosol]);
    f_rfother = convert(Array, df[:rf_other]);
    f_n2oconcentration = convert(Array, rcp_data_conc[:N2O])

    f_foss_hist = convert(Array, Foss_Hist[:EFOSS])

    # Timesteps
    deltat = 1.0
    #Pre-industrial CH4 concentrations

    anomtable = transpose(f_anomtable)

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
    setparameter(m, :ch4cyclemagicc, :CH4_emissions, f_ch4emissions)
    setparameter(m, :ch4cyclemagicc, :NOX_emissions, f_noxemissions)
    setparameter(m, :ch4cyclemagicc, :CO_emissions, f_coemissions)
    setparameter(m, :ch4cyclemagicc, :NMVOC_emissions, f_nmvocemissions)
    setparameter(m, :ch4cyclemagicc, :CH4_natural, 266.5)
    setparameter(m, :ch4cyclemagicc, :CH4_0, 790.97924)  #Pre-industrial CH4 concentrations (value for 1850 from RCP8.5 concnetration data)

    setparameter(m, :rfch4magicc, :deltat, deltat)
    setparameter(m, :rfch4magicc, :N0, 275.42506)  #taken from 1850 RCP85 data
    setparameter(m, :rfch4magicc, :STRATH2O, 0.15)  #Documentation says 0.15 not 0.05 from GCAM
    setparameter(m, :rfch4magicc, :OZ00CH4, 0.161)
    setparameter(m, :rfch4magicc, :TROZSENS, 0.042)
    setparameter(m, :rfch4magicc, :OZCH4, 5.0)
    setparameter(m, :rfch4magicc, :CH4_0, 790.97924)

    setparameter(m, :rfo3magicc, :deltat, deltat)
    setparameter(m, :rfo3magicc, :NOX_emissions, f_noxemissions)
    setparameter(m, :rfo3magicc, :CO_emissions, f_coemissions)
    setparameter(m, :rfo3magicc, :NMVOC_emissions, f_nmvocemissions)
    setparameter(m, :rfo3magicc, :OZNOX, 0.125)
    setparameter(m, :rfo3magicc, :OZCO, 0.0011)
    setparameter(m, :rfo3magicc, :OZVOC, 0.0033)
    setparameter(m, :rfo3magicc, :OZ00CH4, 0.161)
    setparameter(m, :rfo3magicc, :FOSSHIST, f_foss_hist)
    setparameter(m, :rfo3magicc, :TROZSENS, 0.042)
    
    setparameter(m, :rfco2, :a₁, -2.4e-7)
    setparameter(m, :rfco2, :b₁, 7.2e-4)
    setparameter(m, :rfco2, :c₁, -2.1e-4)
    setparameter(m, :rfco2, :CO₂_0, 280.)
    setparameter(m, :rfco2, :N₂O_0, 275.42506)  #taken from 1850 RCP85 data
    setparameter(m, :rfco2, :N₂O, f_n2oconcentration)
    
    setparameter(m, :radiativeforcing, :deltat, deltat)
    setparameter(m, :radiativeforcing, :rf_aerosol, f_rfaerosol)
    setparameter(m, :radiativeforcing, :rf_other, f_rfother)
    setparameter(m, :radiativeforcing, :alpha, 1.)

    setparameter(m, :doeclim, :t2co, 2.0)
    setparameter(m, :doeclim, :kappa, 1.1)
    setparameter(m, :doeclim, :deltat, deltat)
    setparameter(m, :doeclim, :T0,     0.0)
    setparameter(m, :doeclim, :H0,     0.0)

    setparameter(m, :ccm, :deltat, deltat)
    setparameter(m, :ccm, :Q10, 1.311)
    setparameter(m, :ccm, :Beta, 0.502)
    setparameter(m, :ccm, :Eta, 17.7)
    setparameter(m, :ccm, :atmco20, 280.)
    setparameter(m, :ccm, :CO2_emissions, f_co2emissions)
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
    connectparameter(m, :ccm, :emeth, :ch4cyclemagicc, :emeth)

    return m
end
