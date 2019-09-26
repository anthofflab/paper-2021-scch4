########################################################################
# loaddata() : Load Calibration Observational Data
#     model_start = Year to start running model.
#     model_end   = Final year to run model.
########################################################################
function load_calibration_data(model_start::Int64, model_end::Int64)

    #Initialize with years
    df = DataFrame(year = collect(1765:2018))

    #---------------------------------------------------------------------------------
    # HadCRUT4 temperature data (anomalies relative to 1861-1880 mean).
    #---------------------------------------------------------------------------------
    index_1861, index_1880 = findin(collect(1850:2017), [1861, 1880])
#    raw_data = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/global_temp_hadcrut4.csv"), header=25)
    raw_data = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/global_temp_hadcrut4.csv"), header=25)
    dat=DataFrame(year=Array(raw_data[:year]), hadcrut_temperature = Array(raw_data[:median]) .- mean(Array(raw_data[index_1861:index_1880, :median])))
    #dat=DataFrame(year=Array(raw_data[:year]), hadcrut_temperature = Array(raw_data[:median] .- raw_data[1, :median]))
    df = join(df, dat, on=:year, kind=:outer)

    # Read in HadCRUT4 1σ errors
    dat = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/HadCRUT.4.6.0.0.annual_ns_avg.1_(1sigma_uncertainty).csv"), header=22)
    dat = DataFrame(year=Array(dat[:year]), hadcrut_err=Array(dat[:one_sigma_all]))
    df = join(df, dat, on=:year, kind=:outer)
    #---------------------------------------------------------------------------------
    # annual global ocean heat content (0-3000 m)
    # Gouretski & Koltermann, "How much is the ocean really warming?",
    # Geophysical Research Letters 34, L01610 (2007).
    #---------------------------------------------------------------------------------
    #dat = readtable(joinpath(dirname(@__FILE__),"data/new_mcmc_data/gouretski_ocean_heat_3000m.txt"), header=false, separator=' ', names=[:year, :obs_ocheat, :obs_ocheatsigma], commentmark='%', allowcomments=true)
    dat = readtable(joinpath(dirname(@__FILE__),"data/new_mcmc_data/gouretski_ocean_heat_3000m.txt"), header=false, separator=' ', names=[:year, :obs_ocheat, :obs_ocheatsigma], commentmark='%', allowcomments=true)
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # Mauna Loa instrumental CO2
    #---------------------------------------------------------------------------------
    #dat = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/co2_mauna_loa.csv"), header=60)
    dat = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/co2_mauna_loa.csv"), header=60)
    dat = DataFrame(year=Array(dat[:year]), obs_co2inst = Array(dat[:mean]), co2inst_sigma = Array(dat[:unc]))
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # Law Dome ice core CO2
    #---------------------------------------------------------------------------------
    #dat = readtable(joinpath(dirname(@__FILE__),"data/new_mcmc_data/co2iceobs.txt"), separator=' ', header=false, names=[:year, :obs_co2ice, :unknown])
    dat = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/law_dome_co2.csv"), header=5)
    dat = DataFrame(year=Array(dat[:year]), obs_co2ice=Array(dat[:co2_ice]), co2ice_sigma=Array(dat[:one_sigma_error]))
    df = join(df, dat, on=:year, kind=:outer)
    # obs.co2ice.err = rep(4, length(obs.co2ice))

    #---------------------------------------------------------------------------------
    # decadal ocean carbon fluxes, McNeil et al. (2003)
    #---------------------------------------------------------------------------------
    dat = DataFrame(year=[1985, 1995], obs_ocflux=[1.6, 2.0], obs_ocflux_err=[0.4, 0.4])
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # NOAA global CH₄ flask concentration data (http://www.esrl.noaa.gov/gmd/ccgg/trends_ch4/#global_data).
    #---------------------------------------------------------------------------------
    #ch4_obs = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_noaa.csv"), header=59)
    ch4_obs = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_noaa.csv"), header=59)
    dat = DataFrame(year=Array(ch4_obs[:year]), obs_ch4inst=Array(ch4_obs[:mean]), ch4inst_sigma=Array(ch4_obs[:unc]))
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # CH4 ICE CORE DATA
    #---------------------------------------------------------------------------------

    # CH₄ ice core (etheridge 1998).
    #ch4_ice_core = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_ice_core_etheridge1998.csv"), header=15)
    ch4_ice_core = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_ice_core_etheridge1998.csv"), header=15)

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc(Int64, Array(ch4_ice_core[:,:air_age]))
    unique_years = unique(years)

    #Create array to hold annual CH₄ values.
    n_years = length(unique_years)
    ch4_ice_core_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean CH₄ value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findin(years, unique_years[t])
        #Assign year and mean value for array with one value per year.
        ch4_ice_core_avg[t,:year] = unique_years[t]
        ch4_ice_core_avg[t,:ch4] = mean(Array(ch4_ice_core[index, :CH4]))
    end

    # Load CH₄ ice firn data (etheridge 1998).
    #ch4_firn = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_firn_etheridge1998.csv"), header=15)
    ch4_firn = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_firn_etheridge1998.csv"), header=15)

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc(Int64, Array(ch4_firn[:,:air_age]))
    unique_years = unique(years)

    #Create array to hold new values.
    n_years = length(unique_years)
    ch4_firn_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findin(years, unique_years[t])
        #Assign year and mean value for array with one value per year.
        ch4_firn_avg[t,:year] = unique_years[t]
        ch4_firn_avg[t,:ch4] = mean(Array(ch4_firn[index, :CH4]))
    end

    # Merge data for ice core (1852-1977) and firn (1978-1981). After that use NOAA flask data.
    core_index = findin(ch4_ice_core_avg[:,:year], [1852,1977])
    firn_index = findin(ch4_firn_avg[:,:year], [1978,1981])
    ch4_ice = vcat(ch4_ice_core_avg[core_index[1]:core_index[2],:], ch4_firn_avg[firn_index[1]:firn_index[2],:])

    # Do a linear interpolation between interpolar (N-S) differences as described in Etheridge et al. (1998)
    # Uses interpolar difference of 50 ppb for late 1800s, and 143 ppb for 1990s (Etheridge 1998 uses 1992 as end year).
    start_year = 1842
    end_year   = 1985
    n_years    = length(start_year:end_year)

    # Initialize array to hold merged, re-scaled global CH₄ data.
    interp_vals = zeros(n_years, 2)
    interp_vals[:,1] = collect(start_year:end_year)

    # Set initial value.
    interp_vals[1,2] = 41.9
    #Calculate annual amount to add for each year.
    adder = (143-41.9) / (length(start_year:end_year)-1)

    for t in 2:n_years
        interp_vals[t,2] = interp_vals[t-1,2] + adder
    end

    #Find indices of interpolated interpolar differences that correspond to ice core CH₄ observations.
    interp_index = findin(interp_vals[:,1], ch4_ice[:year])

    # Approximate global CH4 concentration as Antartica ice core values plus 37% of interpolar difference (see Etheridge et al. (1998).
    ch4_ice[:global_ch4] = ch4_ice[:ch4] .+ interp_vals[interp_index, 2] .* 0.37

    # Set observation error for CH₄ ice core data as 15 ppb.
    # Etheridge notes 5ppb (1σ) measurement error, 10 ppb (1σ) error in interpolar difference, and global calculation (using interhemispheric mixing assumptions) would introduce an error of at most ≈ 10ppb.
    # Merge re-scaled global CH₄ observations to rest of calibration data.
    ch4ice_error = ones(size(ch4_ice)[1]) .* 15.0
    dat = DataFrame(year=Array(ch4_ice[:year]),  obs_ch4ice = Array(ch4_ice[:global_ch4]), ch4ice_sigma = ch4ice_error)
    df = join(df, dat, on=:year, kind=:outer)

    # Sort observations by year.
    sort!(df, cols=[:year])

    # Find start and end indices
    start_index, end_index = findin(collect(1765:2018), [model_start, model_end])

    return df[start_index:end_index, :]
end


#=
########################################################################
# loaddata() : Load Calibration Observational Data
########################################################################
function loaddata()

    #---------------------------------------------------------------------------------
    # HadCRUT4 temperature data (anomalies relative to 1850-1870 mean).
    #---------------------------------------------------------------------------------
    raw_data = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/global_temp_hadcrut4.csv"), header=25)
    dat=DataFrame(year=Array(raw_data[:year]), obs_temperature = Array(raw_data[:median]) .- mean(Array(raw_data[1:20, :median])))
    df = dat

    #---------------------------------------------------------------------------------
    # annual global ocean heat content (0-3000 m)
    # Gouretski & Koltermann, "How much is the ocean really warming?",
    # Geophysical Research Letters 34, L01610 (2007).
    #---------------------------------------------------------------------------------
    dat = readtable(joinpath(dirname(@__FILE__),"data/new_mcmc_data/gouretski_ocean_heat_3000m.txt"), header=false, separator=' ', names=[:year, :obs_ocheat, :obs_ocheatsigma], commentmark='%', allowcomments=true)
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # Mauna Loa instrumental CO2
    #---------------------------------------------------------------------------------
    dat = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/co2_mauna_loa.csv"), header=60)
    dat = DataFrame(year=Array(dat[:year]), obs_co2inst = Array(dat[:mean]))
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # Law Dome ice core CO2
    #---------------------------------------------------------------------------------
    dat = readtable(joinpath(dirname(@__FILE__),"data/new_mcmc_data/co2iceobs.txt"), separator=' ', header=false, names=[:year, :obs_co2ice, :unknown])
    dat = DataFrame(year=dat[:year], obs_co2ice=dat[:obs_co2ice])
    df = join(df, dat, on=:year, kind=:outer)
    # obs.co2ice.err = rep(4, length(obs.co2ice))

    #---------------------------------------------------------------------------------
    # decadal ocean carbon fluxes, McNeil et al. (2003)
    #---------------------------------------------------------------------------------
    dat = DataFrame(year=[1985, 1995], obs_ocflux=[1.6, 2.0], obs_ocflux_err=[0.4, 0.4])
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # NOAA global CH₄ flask concentration data (http://www.esrl.noaa.gov/gmd/ccgg/trends_ch4/#global_data).
    #---------------------------------------------------------------------------------
    ch4_obs = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_noaa.csv"), header=59)
    dat = DataFrame(year=Array(ch4_obs[:year]), obs_ch4inst=Array(ch4_obs[:mean]))
    df = join(df, dat, on=:year, kind=:outer)

    #---------------------------------------------------------------------------------
    # CH4 ICE CORE DATA
    #---------------------------------------------------------------------------------

    # CH₄ ice core (etheridge 1998).
      ch4_ice_core = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_ice_core_etheridge1998.csv"), header=15)

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc(Int64, Array(ch4_ice_core[:,:air_age]))
    unique_years = unique(years)

    #Create array to hold annual CH₄ values.
    n_years = length(unique_years)
    ch4_ice_core_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean CH₄ value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findin(years, unique_years[t])
        #Assign year and mean value for array with one value per year.
        ch4_ice_core_avg[t,:year] = unique_years[t]
        ch4_ice_core_avg[t,:ch4] = mean(Array(ch4_ice_core[index, :CH4]))
    end

    # Load CH₄ ice firn data (etheridge 1998).
    ch4_firn = CSV.read(joinpath(dirname(@__FILE__),"data/new_mcmc_data/ch4_firn_etheridge1998.csv"), header=15)

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc(Int64, Array(ch4_firn[:,:air_age]))
    unique_years = unique(years)

    #Create array to hold new values.
    n_years = length(unique_years)
    ch4_firn_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findin(years, unique_years[t])
        #Assign year and mean value for array with one value per year.
        ch4_firn_avg[t,:year] = unique_years[t]
        ch4_firn_avg[t,:ch4] = mean(Array(ch4_firn[index, :CH4]))
    end

    # Merge data for ice core (1852-1977) and firn (1978-1981). After that use NOAA flask data.
    core_index = findin(ch4_ice_core_avg[:,:year], [1852,1977])
    firn_index = findin(ch4_firn_avg[:,:year], [1978,1981])
    ch4_ice = vcat(ch4_ice_core_avg[core_index[1]:core_index[2],:], ch4_firn_avg[firn_index[1]:firn_index[2],:])

    # Do a linear interpolation between interpolar (N-S) differences as described in Etheridge et al. (1998)
    # Uses interpolar difference of 50 ppb for late 1800s, and 143 ppb for 1990s (Etheridge 1998 uses 1992 as end year).
    start_year = 1852
    end_year   = 1992
    n_years    = length(start_year:end_year)

    # Initialize array to hold merged, re-scaled global CH₄ data.
    interp_vals = zeros(n_years, 2)
    interp_vals[:,1] = collect(1852:1992)

    # Set initial value.
    interp_vals[1,2] = 50.0
    #Calculate annual amount to add for each year.
    adder = (143-50) / (length(start_year:end_year)-1)

    for t in 2:n_years
        interp_vals[t,2] = interp_vals[t-1,2] + adder
    end

    #Find indices of interpolated interpolar differences that correspond to ice core CH₄ observations.
    interp_index = findin(interp_vals[:,1], ch4_ice[:year])

    # Approximate global CH4 concentration as Antartica ice core values plus 37% of interpolar difference (see Etheridge et al. (1998).
    ch4_ice[:global_ch4] = ch4_ice[:ch4] .+ interp_vals[interp_index, 2] .* 0.37

    # Merge re-scaled global CH₄ observations to rest of calibration data.
    dat = DataFrame(year=Array(ch4_ice[:year]),  obs_ch4ice = Array(ch4_ice[:global_ch4]))
    df = join(df, dat, on=:year, kind=:outer)

    # Sort observations by year.
    sort!(df, cols=[:year])

    return df
end

=#


################################################################################
# ch4_indices() : Calculate AR(1) and i.i.d. indices for CH₄ Ice Core Data.
################################################################################
function ch4_indices(ch4_data)

    # Create empty arrays to hold data indices plus start, end, and total indices for AR(1) blocks.
    data_indices = Array{Int64,1}()
    start_ar1    = Array{Int64,1}()
    end_ar1      = Array{Int64,1}()
    ar1_indices  = Array{Int64,1}()

    # Calculate CH₄ ice core indices that have data.
    for i=1:length(ch4_data)
        if !ismissing(ch4_data[i])
            push!(data_indices, i)
        end
    end

    # Calculate differences in the indices indicating where there are data points.
    diffs = data_indices[2:end] .- data_indices[1:end-1]

    # Calculate all of the indices indicating the start of an AR(1) block.
    for i = 1:length(diffs)
        if i ==1
            # Case where first period starts an AR(1) block.
            if diffs[i] == 1
                push!(start_ar1, data_indices[i])
            end
        else
            if diffs[i-1] != 1 && diffs[i] == 1
                push!(start_ar1, data_indices[i])
            end
        end
    end

    # Calculate all of the indices indicating the end of an AR(1) block.
    for i = 1:length(diffs)
        if i != length(diffs)
            if diffs[i] == 1 && diffs[i+1] != 1
                push!(end_ar1, data_indices[i+1])
            end
        else
            # Case where final period ends an AR(1) block.
            if diffs[i] == 1
                push!(end_ar1, data_indices[i+1])
            end
        end
    end

    # Calculate all of the indices indicating i.i.d. data by filling in AR1 block indices.
    for i = 1:length(start_ar1)
        ar1_indices = vcat(ar1_indices, collect(start_ar1[i]:end_ar1[i]))
    end

    # Calculate iid index as anything not between AR(1) block start and endpoints.
    iid_indices = setdiff(data_indices, ar1_indices)

    return start_ar1, end_ar1, ar1_indices, iid_indices
end
