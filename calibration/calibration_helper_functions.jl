########################################################################
# loaddata() : Load Calibration Observational Data
#     model_end   = Final year to run model (by default, models start in 1765)
########################################################################
function load_calibration_data(model_start::Int, model_end::Int)

    # Set model start year to 1765 by default.
    model_start = 1765

    # Create column of calibration years and calculate indicies for calibration time period (relative to 1768-2018).
    df = DataFrame(year = collect(1765:2018))
    model_calibration_indices = findall((in)(collect(model_start:model_end)), collect(1765:2018))


    #-------------------------------------------------------------------
    # HadCRUT4 temperature data (anomalies relative to 1861-1880 mean).
    #-------------------------------------------------------------------

    raw_temp_data = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "global_temp_hadcrut4.csv"), skiplines_begin=24))

    # Find indices to normalize temperature data to 1861-1880 mean.
    hadcrut_norm_indices = findall((in)(collect(1861:1880)), collect(1850:2017))

    # Normalize temperature data to 1861-1880 mean.
    norm_temp_data  = DataFrame(year=raw_temp_data[!,:year], hadcrut_temperature_obs = raw_temp_data[!,:median] .- mean(raw_temp_data[hadcrut_norm_indices, :median]))

    # Join data on year.
    df = join(df, norm_temp_data, on=:year, kind=:outer)

    # Read in HadCRUT4 1σ errors and rename column.
    raw_temp_errors  = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "global_temp_hadcrut4_1sigma_uncertainty.csv"), skiplines_begin=21))
    rename!(raw_temp_errors, :one_sigma_all => :hadcrut_temperature_sigma)

    # Join data on year
    df = join(df, raw_temp_errors[!, [:year, :hadcrut_temperature_sigma]], on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Annual Global Ocean Heat Content (0-3000 m)
    #---------------------------------------------------------------------------------

    # Load ocean heat content (0-3000m) observations and errors.
    ocean_heat_raw = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ocean_heat_gouretski_3000m.csv"), colnames=["year", "ocean_heat_obs", "ocean_heat_sigma"], skiplines_begin=3))

    # Join data on year.
    df = join(df, ocean_heat_raw, on=:year, kind=:outer)


    #--------------------------------------------------------
    # Mauna Loa Instrumental Atmospheric CO₂ Concentrations.
    #--------------------------------------------------------

    # Load Mauna Loa CO₂ observations and errors, and rename columns.
    raw_mauna_loa_co2_data  = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "co2_mauna_loa.csv"), skiplines_begin=59))
    rename!(raw_mauna_loa_co2_data, :mean => :maunaloa_co2_obs, :unc => :maunaloa_co2_sigma)

    # Join data on year.
    df = join(df, raw_mauna_loa_co2_data, on=:year, kind=:outer)


    #-----------------------------------------------------
    # Law Dome Ice Core Atmospheric CO₂ Concentrations.
    #-----------------------------------------------------

    # Load Law Dome CO₂ observations and errors, and rename columns.
    raw_law_dome_co2_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "law_dome_co2.csv"), skiplines_begin=4))
    rename!(raw_law_dome_co2_data, :co2_ice => :lawdome_co2_obs, :one_sigma_error => :lawdome_co2_sigma)

    # Join data on year.
    df = join(df, raw_law_dome_co2_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Decadal Ocean Carbon Fluxes
    #---------------------------------------------------------------------------------

    # Observations and errors from McNeil et al. (2003).
    ocean_co2_flux_data = DataFrame(year=[1985, 1995], oceanco2_flux_obs=[1.6, 2.0], oceanco2_flux_sigma=[0.4, 0.4])

    # Join data on year.
    df = join(df, ocean_co2_flux_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # NOAA Instrumental Atmospheric CH₄ Concentrations.
    #---------------------------------------------------------------------------------

    # Load NOAA CH₄ observations and errors, and rename columns.
    raw_noaa_ch4_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ch4_noaa.csv"), skiplines_begin=58))
    rename!(raw_noaa_ch4_data, :mean => :noaa_ch4_obs, :unc => :noaa_ch4_sigma)

    # Join data on year.
    df = join(df, raw_noaa_ch4_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Law Dome Ice Core Atmospheric CH₄ Concentrations.
    #---------------------------------------------------------------------------------

    # Load Law Dome CH₄ observations and dates.
    raw_law_dome_ch4_data = DataFrame(load(joinpath(@__DIR__, "..", "data", "calibration_data", "ch4_ice_core_etheridge_1998.csv"), skiplines_begin=14))

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc.(Int, raw_law_dome_ch4_data.air_age)
    unique_years = unique(raw_law_dome_ch4_data.air_age)

    # Create array to hold annual CH₄ values.
    n_years = length(unique_years)
    law_dome_ch4_annual_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean CH₄ value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findall(unique_years[t] .== years)
        #Assign year and mean value for array with one value per year.
        law_dome_ch4_annual_avg[t,:year] = unique_years[t]
        law_dome_ch4_annual_avg[t,:ch4]  = mean(raw_law_dome_ch4_data[index, :CH4])
    end

    # Load CH₄ ice firn observations and dates (also Etheridge 1998).
    raw_ch4_firn_data = DataFrame(load(joinpath(@__DIR__,  "..", "data", "calibration_data", "ch4_firn_etheridge_1998.csv"), skiplines_begin=14))

    # Get years (averaging data together for multiple observations in the same year).
    years = trunc.(Int, raw_ch4_firn_data[:,:air_age])
    unique_years = unique(years)

    #Create array to hold annual CH₄ ice firn values.
    n_years = length(unique_years)
    firn_ch4_annual_avg = DataFrame(year=zeros(Int64, n_years), ch4=zeros(n_years))

    #Calculate mean value for each unique year.
    for t in 1:n_years
        #Find indices where there are multiple observations across a single year.
        index = findall(unique_years[t] .== years)
        #Assign year and mean value for array with one value per year.
        firn_ch4_annual_avg[t,:year] = unique_years[t]
        firn_ch4_annual_avg[t,:ch4] = mean(raw_ch4_firn_data[index, :CH4])
    end

    # Merge data for ice core (1852-1977) and firn (1978-1981) that occur during calibration period. After that, calibration uses NOAA flask data.
    core_indices = findall((in)(collect(1852:1977)), law_dome_ch4_annual_avg[:,:year])
    firn_indices = findall((in)(collect(1978:1981)), firn_ch4_annual_avg[:,:year])
    ch4_ice = vcat(law_dome_ch4_annual_avg[core_indices,:], firn_ch4_annual_avg[firn_indices,:])

    # Do a linear interpolation between interpolar (N-S) differences as described in Etheridge et al. (1998)
    # Uses reported Greenland-Antarctica difference of ≈ 41.9 ppb for 1842 and 143 ppb for 1980s (assumed centered on 1985).
    start_year = 1842
    end_year   = 1985
    n_years    = length(start_year:end_year)

    # Initialize array to hold merged, re-scaled global CH₄ data.
    interp_vals = zeros(n_years, 2)
    interp_vals[:,1] = collect(start_year:end_year)

    # Set initial value.
    interp_vals[1,2] = 41.9

    #Calculate annual amount to add for each year in interpolation.
    adder = (143-41.9) / (length(start_year:end_year)-1)

    # Calculate interpolar values to be added to ice core data over time if we had annually consecutive observations.
    for t in 2:n_years
        interp_vals[t,2] = interp_vals[t-1,2] + adder
    end

    #Find indices of annual interpolated interpolar differences that correspond to ice core CH₄ observations.
    interp_index = findall((in)(ch4_ice[!,:year]), interp_vals[:,1])

    # Approximate global CH4 concentration as Antartica ice core values plus 37% of interpolar difference (see Etheridge et al. (1998).
    ch4_ice[!,:global_ch4] = ch4_ice[!,:ch4] .+ interp_vals[interp_index, 2] .* 0.37

    # Set observation error for CH₄ ice core data as 15 ppb.
    # Etheridge notes 5ppb (1σ) measurement error and global calculation (using interhemispheric mixing assumptions) could introduce an error of ≈ 10ppb.
    ch4ice_error = ones(size(ch4_ice)[1]) .* 15.0
    final_law_dome_ch4_data = DataFrame(year=ch4_ice[!,:year],  lawdome_ch4_obs = ch4_ice[!,:global_ch4], lawdome_ch4_sigma = ch4ice_error)

    # Join re-scaled global CH₄ observations with rest of calibration data.
    df = join(df, final_law_dome_ch4_data, on=:year, kind=:outer)


    #---------------------------------------------------------------------------------
    # Finalize Joint Calibration Data Set.
    #---------------------------------------------------------------------------------

    # Sort all calibration data by year.
    sort!(df, cols=[:year])

    # Crop data to appropriate calibration years and return.
    return df[model_calibration_indices, :]
end



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
