include("utils/helper_functions.jl")
include("calibration/calibration_helper_functions.jl")
include("calibration/create_log_posterior.jl")
include("calibration/create_log_posterior_wide_priors.jl")

include("create_models/additional_model_components/rf_ch4_direct_fund.jl")
include("create_models/additional_model_components/rf_ch4_etminan.jl")
include("create_models/additional_model_components/rf_ch4_myhre_fair.jl")
include("create_models/additional_model_components/rf_ch4_total_fund.jl")
include("create_models/additional_model_components/rf_co2_etminan.jl")
include("create_models/additional_model_components/total_co2_emissions.jl")
include("create_models/additional_model_components/rf_total.jl")

include("create_models/create_iam_dice.jl")
include("create_models/create_iam_fund.jl")
include("create_models/create_sneasy_fairch4.jl")
include("create_models/create_sneasy_fundch4.jl")
include("create_models/create_sneasy_hectorch4.jl")
include("create_models/create_sneasy_magiccch4.jl")

include("calibration/run_climate_models/run_sneasy_fairch4.jl")
include("calibration/run_climate_models/run_sneasy_fundch4.jl")
include("calibration/run_climate_models/run_sneasy_hectorch4.jl")
include("calibration/run_climate_models/run_sneasy_magiccch4.jl")

include("climate_projections/sneasych4_baseline_case.jl")
include("climate_projections/sneasych4_outdated_forcing.jl")
include("climate_projections/sneasych4_remove_correlations.jl")
include("climate_projections/sneasych4_us_ecs.jl")

include("scch4/dice_scch4.jl")
include("scch4/fund_scch4.jl")
