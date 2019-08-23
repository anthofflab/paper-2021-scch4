
using Mimi
using DataFrames
using DataFramesMeta

include("doeclim.jl")
include("ccm.jl")
include("radiativeforcing.jl")
include("rfco2.jl")
include("ch4cycle_fund.jl")
include("rfch4_fund_oldRF.jl")

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
    addcomponent(m, ch4cyclecomponent.ch4cycle, :ch4cycle)
    addcomponent(m, rfch4component.rfch4, :rfch4)
    addcomponent(m, rfco2component.rfco2, :rfco2)
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
    f_ch4concentration = convert(Array, rcp_data_conc[:CH4])
    f_n2oconcentration = convert(Array, rcp_data_conc[:N2O])

    # Timesteps
    deltat = 1.0
    anomtable = transpose(f_anomtable)

    # ---------------------------------------------
    # Set parameters
    # ---------------------------------------------

    setparameter(m, :doeclim, :t2co, 2.0)
    setparameter(m, :doeclim, :kappa, 1.1)
    setparameter(m, :doeclim, :H0, 0.0)
    setparameter(m, :doeclim, :deltat, deltat)

    setparameter(m, :ccm, :deltat, deltat)
    setparameter(m, :ccm, :Q10, 1.311)
    setparameter(m, :ccm, :Beta, 0.502)
    setparameter(m, :ccm, :Eta, 17.7)
    setparameter(m, :ccm, :atmco20, 280.)
    setparameter(m, :ccm, :CO2_emissions, f_co2emissions)
    setparameter(m, :ccm, :anomtable, anomtable)

    setparameter(m, :rfco2, :a₁, -2.4e-7)
    setparameter(m, :rfco2, :b₁, 7.2e-4)
    setparameter(m, :rfco2, :c₁, -2.1e-4)
    setparameter(m, :rfco2, :CO₂_0, 280.)
    setparameter(m, :rfco2, :N₂O_0, 275.42506)  #taken from 1850 RCP85 data
    setparameter(m, :rfco2, :N₂O, f_n2oconcentration)

    setparameter(m, :ch4cycle, :globch4, f_ch4emissions)
    setparameter(m, :ch4cycle, :lifech4, 12.)
    setparameter(m, :ch4cycle, :ch4pre, 721.89411) #taken from 1765 RCP85 data
    setparameter(m, :ch4cycle, :deltat, deltat)

    setparameter(m, :rfch4, :n2opre, 272.95961) #taken from 1765 RCP85 data
    setparameter(m, :rfch4, :ch4pre, 721.89411) #taken from 1765 RCP85 data
    setparameter(m, :rfch4, :ch4ind, 0.4) #Set to 0.15 because 0.25 is from ch4 effect on O3, but can't separate that out?
    # setparameter(m, :rfch4, :ch4ind, 0.15) 
    setparameter(m, :rfch4, :deltat, deltat)

    setparameter(m, :radiativeforcing, :rf_aerosol, f_rfaerosol)
    setparameter(m, :radiativeforcing, :rf_other, f_rfother)
    setparameter(m, :radiativeforcing, :alpha, 1.)
    setparameter(m, :radiativeforcing, :deltat, deltat)

    # ---------------------------------------------
    # Connect parameters to variables
    # ---------------------------------------------

    connectparameter(m, :doeclim, :forcing, :radiativeforcing, :rf)
    connectparameter(m, :ccm, :temp, :doeclim, :temp)
    connectparameter(m, :rfco2, :CO₂, :ccm, :atmco2)
    connectparameter(m, :rfch4, :acch4, :ch4cycle, :acch4)
    connectparameter(m, :radiativeforcing, :rf_ch4, :rfch4, :rf_ch4)
    connectparameter(m, :radiativeforcing, :rf_co2, :rfco2, :rf_co2)

    return m
end
