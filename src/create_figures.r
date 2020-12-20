# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file replicates the figures from Errickson et al. (2020) using the results from "main.jl"
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------

renv::restore()

# Suppress warnings.
options(warn = -1)

# Load required R packages.
library(data.table)
library(egg)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(msm)
library(scales)

# Name of folder containing replication results produced by "main.jl" file.
results_folder_name = "my_results"

#####################################################################################################
#####################################################################################################
# Run Social Cost of Methane Figure Replication Code
#####################################################################################################
#####################################################################################################

# Load generic helper functions file.
source(file.path("src", "utils", "figure_helper_functions.r"))

# Load calibration observations for hindcasts.
obs = read.csv(file.path("data", "calibration_data", "calibration_data_combined.csv"))

# Set default colors for different versions of SNEASY+CH4.
fairch4_color   = "#10BBFD"
fundch4_color   = "#785EF0"
hectorch4_color = "#DC267F"
magiccch4_color = "#FE6100"

# Set default colors for different SC-CH4 estimation scenarios.
base_color    = "#2bc3a1"
forcing_color = "#2f74ec"
corr_color    = "#079818"
us_ecs_color  = "#cb67e5"

#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Figure 1 - Temperature and Methane Hindcasts.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Load temperature and methane credible interval results for baseline projections.
fair_temperature_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_temperature.csv"))
fund_temperature_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_temperature.csv"))
hector_temperature_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_temperature.csv"))
magicc_temperature_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_temperature.csv"))

fair_ch4_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_ch4.csv"))
fund_ch4_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_ch4.csv"))
hector_ch4_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_ch4.csv"))
magicc_ch4_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_ch4.csv"))

# Load temperature credible interval results for projections using U.S. climate sensitivity values.
fair_temperature_ci_ecs   = read.csv(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_fair", "ci_temperature.csv"))
fund_temperature_ci_ecs   = read.csv(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_fund", "ci_temperature.csv"))
hector_temperature_ci_ecs = read.csv(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_hector", "ci_temperature.csv"))
magicc_temperature_ci_ecs = read.csv(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_magicc", "ci_temperature.csv"))

# Set some common figure settings.
mean_width = 0.3
point_size = 1.0
point_stroke = 0.2
outer_ci_width = 0.2
inner_ci_width = 0.2
plot_years = c(1850, 2022)
point_color = "yellow"
ecs_color = "gray70"

#-------------------------
# Temperature Hindcasts.
#-------------------------

fair_temperature_hindcast   =  climate_projection(fair_temperature_ci_base, obs, "hadcrut_temperature_obs", fairch4_color, c(-0.5,2.03), plot_years, "", "Surface Temperature \n Increase (*C)", c(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0), c("-0.5", "0", "0.5", "1", "1.5", "2.0"), FALSE, TRUE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FAIR", c(1856, 1.82), TRUE, fair_temperature_ci_ecs, ecs_color)
# Override margins to allow room for title
fair_temperature_hindcast   = fair_temperature_hindcast + theme(plot.margin = unit(c(0.1,0.1,0.1,0.4), "cm"))

fund_temperature_hindcast   =  climate_projection(fund_temperature_ci_base, obs, "hadcrut_temperature_obs", fundch4_color, c(-0.5,2.03), plot_years, "", "", c(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0), c("-0.5", "0", "0.5", "1", "1.5", "2.0"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FUND", c(1856, 1.82), TRUE, fund_temperature_ci_ecs, ecs_color)

hector_temperature_hindcast =  climate_projection(hector_temperature_ci_base, obs, "hadcrut_temperature_obs", hectorch4_color, c(-0.5,2.03), plot_years, "", "", c(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0), c("-0.5", "0", "0.5", "1", "1.5", "2.0"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-Hector", c(1856, 1.82), TRUE, hector_temperature_ci_ecs, ecs_color)

magicc_temperature_hindcast =  climate_projection(magicc_temperature_ci_base, obs, "hadcrut_temperature_obs", magiccch4_color, c(-0.5,2.03), plot_years, "", "", c(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0), c("-0.5", "0", "0.5", "1", "1.5", "2.0"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-MAGICC", c(1856, 1.82), TRUE, magicc_temperature_ci_ecs, ecs_color)

#----------------------------------
# Methane Concentration Hindcasts.
#----------------------------------

fair_ch4_hindcast   = climate_projection(fair_ch4_ci_base, obs, "noaa_ch4_obs", fairch4_color, c(0, 2200), plot_years, "Year", expression(paste("           Methane \n Concentration (ppb)")), c(0, 500, 1000, 1500, 2000), c("0", "500", "1000", "1500", "2000"), TRUE, TRUE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FAIR", c(1855, 2017), FALSE, NULL, NULL)
fair_ch4_hindcast   = fair_ch4_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_ch4_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)
fair_ch4_hindcast   = fair_ch4_hindcast + theme(plot.margin=unit(c(0.1,0.1,0.1,0.4), "cm"))

fund_ch4_hindcast   = climate_projection(fund_ch4_ci_base, obs, "noaa_ch4_obs", fundch4_color, c(0,2200), plot_years, "Year", "", c(0, 500, 1000, 1500, 2000), c("0", "500", "1000", "1500", "2000"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FUND", c(1855, 2017), FALSE, NULL, NULL)
fund_ch4_hindcast   = fund_ch4_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_ch4_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

hector_ch4_hindcast = climate_projection(hector_ch4_ci_base, obs, "noaa_ch4_obs", hectorch4_color, c(0,2200), plot_years, "Year", "", c(0, 500, 1000, 1500, 2000), c("0", "500", "1000", "1500", "2000"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-Hector", c(1855, 2017), FALSE, NULL, NULL)
hector_ch4_hindcast = hector_ch4_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_ch4_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

magicc_ch4_hindcast = climate_projection(magicc_ch4_ci_base, obs, "noaa_ch4_obs", magiccch4_color, c(0,2200), plot_years, "Year", "", c(0, 500, 1000, 1500, 2000), c("0", "500", "1000", "1500", "2000"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-MAGICC", c(1855, 2017), FALSE, NULL, NULL)
magicc_ch4_hindcast = magicc_ch4_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_ch4_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

# Arranage individual hindcast figures into panel.
panel_labels   = c("a","b","c","d", "e", "f", "g", "h")
label_design   = list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=1.1, hjust=c(-0.2, 0.2, 0.2, 0.2, -0.2, 0.2, 0.2, 0.2))

fig_1 = ggarrange(fair_temperature_hindcast, fund_temperature_hindcast, hector_temperature_hindcast, magicc_temperature_hindcast,
	   		      fair_ch4_hindcast, fund_ch4_hindcast, hector_ch4_hindcast, magicc_ch4_hindcast,
				  nrow=2, ncol=4, labels=panel_labels, label.args = label_design)

# Save a .jpg and .pdf version of Figure 1.
ggsave(fig_1, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Figure_1.jpg"), device="jpeg", type="cairo", width=183, height=70, unit="mm", dpi=300)
ggsave(fig_1, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Figure_1.pdf"), device="pdf", width=183, height=70, unit="mm", useDingbats = FALSE)



#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Figure 2 - Various SC-CH4 Distributions.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Load baseline SC-CH4 results for DICE.
dice_scch4_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)
dice_scch4_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)
dice_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)
dice_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load baseline SC-CH4 results for FUND.
fund_scch4_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)
fund_scch4_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)
fund_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)
fund_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load different scenario SC-CH4 estimates for FUND + SNEASY-MAGICC.
fund_scch4_base   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_base) = "scch4"
fund_scch4_oldrf  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_oldrf) = "scch4"
fund_scch4_nocorr = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_nocorr) = "scch4"
fund_scch4_ecs    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_ecs) = "scch4"

# Load different scenario SC-CH4 estimates for DICE + SNEASY-MAGICC.
dice_scch4_base   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_base) = "scch4"
dice_scch4_oldrf  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_oldrf) = "scch4"
dice_scch4_nocorr = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_nocorr) = "scch4"
dice_scch4_ecs    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_ecs) = "scch4"

#-------------------------------
# Baseline SC-CH4 Distributions
#-------------------------------

# Merge baseline data into a data.frame for plotting.
scch4_fairch4   = data.frame(fund=fund_scch4_fairch4[,1], dice=dice_scch4_fairch4[,1])
scch4_fundch4   = data.frame(fund=fund_scch4_fundch4[,1], dice=dice_scch4_fundch4[,1])
scch4_hectorch4 = data.frame(fund=fund_scch4_hectorch4[,1], dice=dice_scch4_hectorch4[,1])
scch4_magiccch4 = data.frame(fund=fund_scch4_magiccch4[,1], dice=dice_scch4_magiccch4[,1])

# Calculate baseline SC-CH4 means for each IAM.
base_dice_mean = mean(c(dice_scch4_fairch4[,1], dice_scch4_fundch4[,1], dice_scch4_hectorch4[,1], dice_scch4_magiccch4[,1]))
base_fund_mean = mean(c(fund_scch4_fairch4[,1], fund_scch4_fundch4[,1], fund_scch4_hectorch4[,1], fund_scch4_magiccch4[,1]))

# Create a vector of alpha values to set color transparency levels.
alphas = rep(1.0, 4)

# Order colors in terms of increasing SC-CH4 estimates by model.
scch4_colors = c(fundch4_color, hectorch4_color, fairch4_color, magiccch4_color)

# Create Figure 2a and add points identifying each IAM's mean SC-CH4 estimate.
fig_2a = scch4_pdf_baseline(scch4_fundch4, scch4_hectorch4, scch4_fairch4, scch4_magiccch4, alphas, scch4_colors, 0.4, c(-100,3300), c(0,500,1000,1500,2000,2500,3000), c("0","500","1000","1500","2000","2500","3000"), "Social Cost of Methane ($/t-CH4)", c(0,0.00255))
fig_2a = fig_2a + geom_point(aes(x=c(base_fund_mean, base_dice_mean), y=c(0,0)), shape=c(21,23), size=2.25,  fill="white", stroke=0.3)

#-------------------------------
# SC-CH4 Scenario Distributions
#-------------------------------

# Set alpha values and order of different scenario colors.
scenario_colors = c(forcing_color, base_color, corr_color, us_ecs_color)

# Calculate baseline SC-CH4 means for each scenario - IAM pair.
scenario_mean_data = data.frame(zeros = c(0,0,0,0),
                                fund  = c(mean(fund_scch4_oldrf[,1]), mean(fund_scch4_base[,1]), mean(fund_scch4_nocorr[,1]), mean(fund_scch4_ecs[,1])),
                                dice  = c(mean(dice_scch4_oldrf[,1]), mean(dice_scch4_base[,1]), mean(dice_scch4_nocorr[,1]), mean(dice_scch4_ecs[,1])))

# Create Figure 2b plots for FUND.
fig_2b_top = scch4_pdf_scenario(fund_scch4_oldrf, fund_scch4_base, fund_scch4_nocorr, fund_scch4_ecs, alphas, scenario_colors, 0.3, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "", FALSE, c(0,0.0026), "solid", c(0.5,0.2,0.2,0))
fig_2b_top = fig_2b_top + geom_point(data=scenario_mean_data, aes_string(x="fund", y="zeros"), shape=21, size=1.4, fill=scenario_colors, stroke=0.17)

# Create Figure 2b plots for DICE.
fig_2b_bottom = scch4_pdf_scenario(dice_scch4_oldrf, dice_scch4_base, dice_scch4_nocorr, dice_scch4_ecs, alphas, scenario_colors, 0.3, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "Social Cost of Methane ($/t-CH4)", TRUE, c(0,0.0026), "22", c(-2.2,0.2,0.2,0))
fig_2b_bottom = fig_2b_bottom + geom_point(data=scenario_mean_data, aes_string(x="dice", y="zeros"), shape=23, size=1.4, fill=scenario_colors, stroke=0.17)

# Set panel design.
fig_2a_design = list(gp = grid::gpar(fontsize = 8, fontface="bold"), hjust=-5.0, vjust=1.5)
fig_2b_design = list(gp = grid::gpar(fontsize = 8, fontface="bold"), hjust=0.0, vjust=1.5)

# Arrange all Figure 2 panels together.
fig_2a_panel = ggarrange(fig_2a, labels=c("a"), label.args=fig_2a_design)
fig_2b_panel = ggarrange(fig_2b_top, fig_2b_bottom, ncol=1, labels=c("b",""), label.args = fig_2b_design)
fig_2        = grid.arrange(fig_2a_panel, fig_2b_panel, widths=c(3,2.5), nrow=1, ncol=2)

# Save a .jpg and .pdf version of Figure 2.
ggsave(fig_2, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Figure_2.jpg"), device="jpeg", type="cairo", width=136, height=70, unit="mm", dpi=300)
ggsave(fig_2, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Figure_2.pdf"), device="pdf", width=136, height=70, unit="mm", useDingbats = FALSE)



#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Figure 3 - Posterior Parameter and SC-CH4 Correlations.
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Load S-MAGICC temperature projections and confidence intervals (baseline case and scenario without posterior correlations).
magicc_temperature_baseline    = fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "base_temperature.csv"), data.table=FALSE)
magicc_temperature_no_corr     = fread(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_magicc", "base_temperature.csv"), data.table=FALSE)
magicc_temperature_ci_baseline = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_temperature.csv"))
magicc_temperature_ci_no_corr  = read.csv(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_magicc", "ci_temperature.csv"))

# Load S-MAGICC posterior parameters and corresponding SC-CH4 estimates for scatter plots.
post_param_magiccch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_magicc", "parameters_100k.csv"), data.table=FALSE)
dice_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)
fund_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)

#-------------------------------
# Temperature Projection
#-------------------------------

# Settings for projection plot
mean_width = 0.4
ci_width = 0.25
point_size = 1.0
point_stroke = 0.2
no_corr_color = "gray80"

# Plot temperature projections (with and without posterior correlations).
fig_3a_main =  climate_projection(magicc_temperature_ci_baseline, obs, "hadcrut_temperature_obs", magiccch4_color, c(-0.5,12.5), c(1850,2210), "Year", "Surface Temperature (C)", seq(0,12,by=2), as.character(seq(0,12,by=2)), TRUE, TRUE, point_size, point_stroke, ci_width, "dashed", mean_width, "", c(1856, 15.5), TRUE, magicc_temperature_ci_no_corr, no_corr_color)
fig_3a_main = fig_3a_main + theme(plot.margin = unit(c(3,1,5,1), "mm"))

# Get years for temperature pdf inset.
sneasy_years = 1765:2300
pdf_years = c(2050, 2100)

# Isolate baseline projections.
baseline_pdf_data = magicc_temperature_baseline[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]
colnames(baseline_pdf_data) = c("Year_1", "Year_2")

# Isolate no correlation projections.
no_corr_pdf_data = magicc_temperature_no_corr[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]
colnames(no_corr_pdf_data) = c("Year_1", "Year_2")

# Create temperature pdf inset.
fig_3a_inset = ggarrange(inset_nocorr_pdfs(baseline_pdf_data, no_corr_pdf_data, c(magiccch4_color, "gray75"), c(0.8,0.6), 0.2, c(0,8), c(0,2,4,6,8), c("0","2","4","6","8"), "Surface Temperature (C)", c(0,1.71), c(0,0,0,0)), nrow=1, ncol=1)

# Join Fig 3a with inset
fig_3a = fig_3a_main + annotation_custom(grob=fig_3a_inset, xmin=1860, xmax=2045, ymin=3.5, ymax=10.5)

#-----------------------------------
# Parameter & SC-CH4 Scatter Plots
#-----------------------------------

# Set an upper value on point size to create a "max_val+" bin for graph legibiity (99th quantile, rounded to next integer).
upper_bound_size = ceiling(as.numeric(quantile(post_param_magiccch4$Q10, 0.99)))
post_param_magiccch4$Q10_size = post_param_magiccch4$Q10
post_param_magiccch4[which(post_param_magiccch4$Q10 > upper_bound_size), "Q10_size"] = upper_bound_size

# Create plot data dataframe.
scatter_data = data.frame(ECS=post_param_magiccch4$ECS, aerosol=post_param_magiccch4$alpha, Q10_size = post_param_magiccch4$Q10_size, heat_diffusion=post_param_magiccch4$kappa, dice=dice_scch4_magiccch4[,1], fund=fund_scch4_magiccch4[,1])

# Create parameter and SC-CH4 scatter plots.
fig_3b = scatter_4way(scatter_data, c("aerosol", "ECS", "Q10_size", "heat_diffusion"), 5000, 21, 0.5, c("blue", "dodgerblue", "cyan", "yellow"), c(2.0, 4.0), c(2.0, 4.0), c("< 2.0", "> 4.0"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,2.03), c(0,0.5,1,1.5,2),  c("0","0.5","1","1.5","2"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), "Aerosol Radiative Forcing Scale Factor", "Equilibirium Climate Sensitivity (C)", c("Ocean Heat \n Diffusivity", "Carbon Sink Respiration \n Temperature Sensitivity"), c(6,2,3,3), FALSE, "", "", TRUE, TRUE)
fig_3c = scatter_4way(scatter_data, c("ECS", "dice", "Q10_size", "aerosol"), 5000, 23, 0.5, c("yellow", "red", "blue"), c(0.6, 1.5), c(0.6,1.5), c("< 0.6", "> 1.5"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), c(0,3600), c(0,1000,2000,3000), c("0","1000","2000","3000"), "Equilibirium Climate Sensitivity (C)", "Social Cost of Methane ($/t-CH4)", c("Aerosol Forcing \n Scale Factor", "Carbon Sink Respiration \n Temperature Sensitivity"), c(6,3,3,2), TRUE, c("ECS", "fund", "Q10_size", "aerosol"), 21, TRUE, TRUE)

# Arrange all panels into figure 3.
fig_3_top_panel    = ggarrange(fig_3a, labels=c("a"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold", vjust=-3)))
fig_3_bottom_panel = ggarrange(fig_3b, fig_3c, labels=c("b", "c"), nrow=1, ncol=2, label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold")))
fig_3 = grid.arrange(fig_3_top_panel, fig_3_bottom_panel, nrow=2, ncol=1, heights=c(1.5,2))

# Save a .jpg and .pdf version of Figure 3.
ggsave(fig_3, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Figure_3.jpg"), device="jpeg", type="cairo", width=136, height=150, unit="mm", dpi=300)
ggsave(fig_3, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Figure_3.pdf"), device="pdf", width=136, height=150, unit="mm", useDingbats = FALSE)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Figure 4 - Temperature and Discounted Climate Damage Impulse Responses
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Load temperature and discounted damage projections (take transpose for plotting convenience -> row = year, column = new model run).
temperature_base = transpose(fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "base_temperature.csv"), data.table=FALSE))
temperature_pulse = transpose(fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "pulse_temperature.csv"), data.table=FALSE))

# Load DICE marginal discounted damages for different discount rates.
dice_damages_25 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_25.csv"), data.table=FALSE))
dice_damages_30 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_30.csv"), data.table=FALSE))
dice_damages_50 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_50.csv"), data.table=FALSE))
dice_damages_70 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "discounted_damages_70.csv"), data.table=FALSE))

# Load FUND marginal discounted damages for different discount rates.
fund_damages_25 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_25.csv"), data.table=FALSE))
fund_damages_30 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_30.csv"), data.table=FALSE))
fund_damages_50 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_50.csv"), data.table=FALSE))
fund_damages_70 = transpose(fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "discounted_damages_70.csv"), data.table=FALSE))

#-----------------------------------
# Temperature Impulse Response
#-----------------------------------

# Calculate temperature impulse responses.
temperature_response = temperature_pulse - temperature_base

# Calculate mean and 95% interval responses
percentile = 0.95
lower = (1-percentile) / 2
upper =  1 - lower

temperature_response_CI = data.frame(Year = 1765:2300,
                          Mean  = rowMeans(temperature_response),
                          Lower_CI = apply(temperature_response, 1, quantile, lower),
                          Upper_CI = apply(temperature_response, 1, quantile, upper))

# Set years for inset distributions.
sneasy_years = 1765:2300
pdf_years    = c(2030,2040,2050,2070)

# Get mean temperature values for years corresponding to inset distributions.
circle_indices = which(sneasy_years == pdf_years[1] | sneasy_years == pdf_years[2] | sneasy_years == pdf_years[3] | sneasy_years == pdf_years[4])
mean_circle_data = data.frame(x=pdf_years, y=temperature_response_CI$Mean[circle_indices])

# Create temperature impulse response plot.
fig_4a_main = spaghetti_single(temperature_response, temperature_response_CI, "#efab23", 0.15, c(1765, 2300), 2020, 500, c(0,6.5e-11), c(0,2e-11,4e-11,6e-11), c("0", "2e-11", "4e-11", "6e-11"), "Year", "Temperature Response (C/t-CH4)", c(2010,2160), c(2020, 2060, 2100, 2140), c("2020", "2060", "2100", "2140"))

# Add cirlces for years with distribution inset plots.
fig_4a_main = fig_4a_main + geom_point(data=mean_circle_data, aes(x=x, y=y), shape=21, colour="black", fill="black", size=2.0)
fig_4a_main = fig_4a_main + geom_point(data=mean_circle_data, aes(x=x, y=y), shape=21, colour="black", fill=c("#ff7272", "#b67be0", "#86e0a9", "#42c5f4"), size=1.75)

# Isolate data for temperature distribution inset.
temperature_pdf_data = transpose(temperature_response[which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2] | sneasy_years==pdf_years[3] | sneasy_years==pdf_years[4]), ])
colnames(temperature_pdf_data) = c("Year_1", "Year_2", "Year_3", "Year_4")

# Create temperature distribution inset.
fig_4a_inset = inset_4pdfs(temperature_pdf_data, c("#ff7272", "#b67be0", "#86e0a9", "#42c5f4"), rep(0.8,4), rep(1,4), 0.25, c(0,6.5e-11), c(0, 3e-11, 6e-11), c("0.0", "3e-11", "6e-11"), "Temperature", c(0,1.6e11), 6, c(-0.1,0.0,0.0,0.0))

# Add inset to temperature impulse response figure.
fig_4a = fig_4a_main + annotation_custom(grob=ggplotGrob(fig_4a_inset), xmin=2065, xmax=Inf, ymin=(32/75)*6.5e-11, ymax=Inf)

#-----------------------------------
# Climate Damage Impulse Response
#-----------------------------------

# Set damage impulse colors and transparency.
impulse_colors = c("darkorange", "red","dodgerblue", "mediumseagreen")
impulse_alphas = c(0.7,0.3,0.15,0.07)

# Set years for each IAM.
dice_years = 2010:2300
fund_years = 1950:2300

# Get mean response for DICE across different discount rates.
dice_damages_mean = data.frame(Year = 2010:2300,
                               Mean1 = rowMeans(dice_damages_25),
                               Mean2 = rowMeans(dice_damages_30),
                               Mean3 = rowMeans(dice_damages_50),
                               Mean4 = rowMeans(dice_damages_70))

# Get mean response for FUND across different discount rates.
fund_damages_mean = data.frame(Year = 1950:2300,
                               Mean1 = rowMeans(fund_damages_25),
                               Mean2 = rowMeans(fund_damages_30),
                               Mean3 = rowMeans(fund_damages_50),
                               Mean4 = rowMeans(fund_damages_70))

# Create damage impulse response graphs.
fig_4b_main = spaghetti_multi(dice_damages_25, dice_damages_30, dice_damages_50, dice_damages_70, dice_damages_mean, impulse_colors, impulse_alphas, c(2010, 2300), 2020, 250, c(-10,65), c(0, 30, 60), c("0", "30", "60"), "Year", "Damage Response ($/t-CH4)", c(2015,2160), c(2020,2060,2100,2140), c("2020","2060","2100","2140"))
fig_4c_main = spaghetti_multi(fund_damages_25, fund_damages_30, fund_damages_50, fund_damages_70, fund_damages_mean, impulse_colors, impulse_alphas, c(1950, 2300), 2020, 250, c(-10,65), c(0, 30, 60), c("0", "30", "60"), "Year", "Damage Response ($/t-CH4)", c(2015,2160), c(2020,2060,2100,2140), c("2020","2060","2100","2140"))

# Isolate damage distribution inset data for DICE and FUND.
dice_pdf_data_25 = transpose(dice_damages_25[which(dice_years==pdf_years[1] | dice_years==pdf_years[2] | dice_years==pdf_years[3] | dice_years==pdf_years[4]), ])
dice_pdf_data_50 = transpose(dice_damages_50[which(dice_years==pdf_years[1] | dice_years==pdf_years[2] | dice_years==pdf_years[3] | dice_years==pdf_years[4]), ])
colnames(dice_pdf_data_25) = c("Year_1", "Year_2", "Year_3", "Year_4")
colnames(dice_pdf_data_50) = c("Year_1", "Year_2", "Year_3", "Year_4")

fund_pdf_data_25 = transpose(fund_damages_25[which(fund_years==pdf_years[1] | fund_years==pdf_years[2] | fund_years==pdf_years[3] | fund_years==pdf_years[4]), ])
fund_pdf_data_50 = transpose(fund_damages_50[which(fund_years==pdf_years[1] | fund_years==pdf_years[2] | fund_years==pdf_years[3] | fund_years==pdf_years[4]), ])
colnames(fund_pdf_data_25) = c("Year_1", "Year_2", "Year_3", "Year_4")
colnames(fund_pdf_data_50) = c("Year_1", "Year_2", "Year_3", "Year_4")

# Create DICE inset and add to main panel.
dice_25_pdf = inset_4pdfs(dice_pdf_data_25, rep("darkorange",4), c(0.5,0.4,0.35,0.3), c(1,3,2,4), 0.25, c(-1,51), c(0,25,50), c("0","25","50"), "Discounted Damages", c(0,0.29), 6, c(-1.5,0.0,0.0,0.0))
dice_50_pdf = inset_4pdfs(dice_pdf_data_50, rep("dodgerblue",4), c(0.5,0.4,0.35,0.3), c(1,3,2,4), 0.25, c(-1,51), c(0,25,50), c("0","25","50"), NULL, c(0,0.29), 6, c(-0.01,0.0,0.0,0.0))
fig_4b_inset = ggarrange(dice_50_pdf, dice_25_pdf, nrow=2, ncol=1)
fig_4b = fig_4b_main + annotation_custom(grob=fig_4b_inset, xmin=2080, xmax=Inf, ymin=22, ymax=Inf)

# Create FUND inset and add to main panel.
fund_25_pdf = inset_4pdfs(fund_pdf_data_25, rep("darkorange",4), c(0.5,0.4,0.35,0.3), c(1,3,2,4), 0.25, c(-1,20), c(0,10,20), c("0","10","20"), "Discounted Damages", c(0,1.05), 6, c(-1.5,0.0,0.0,0.0))
fund_50_pdf = inset_4pdfs(fund_pdf_data_50, rep("dodgerblue",4), c(0.5,0.4,0.35,0.3), c(1,3,2,4), 0.25, c(-1,20), c(0,10,20), c("0","10","20"), NULL, c(0,1.05), 6, c(-0.01,0.0,0.0,0.0))
fig_4c_inset = ggarrange(fund_50_pdf, fund_25_pdf, nrow=2, ncol=1)
fig_4c = fig_4c_main + annotation_custom(grob=fig_4c_inset, xmin=2080, xmax=Inf, ymin=22, ymax=Inf)

# Set Figure 4 panel design.
label_design   = list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=1.1, hjust=c(0.0, 0.2, 0.2))

# Combine all panels into Figure 4.
figure_4 = ggarrange(fig_4a, fig_4b, fig_4c, nrow=1, ncol=3, labels=c("a","b","c"), label.args = label_design)

# Save a .jpg and .pdf version of Figure 4.
ggsave(figure_4, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Figure_4.jpg"), device="jpeg", type="cairo", width=183, height=70, unit="mm", dpi=300)
ggsave(figure_4, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Figure_4.pdf"), device="pdf", width=183, height=70, unit="mm", useDingbats = FALSE)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Figure 5 - Equity-Weighted SC-CH4 Estimates using FUND.
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Load credible interval results for all FUND regions and eta values.
eq_ci_00 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_00.csv"), data.table=FALSE)
eq_ci_01 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_01.csv"), data.table=FALSE)
eq_ci_02 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_02.csv"), data.table=FALSE)
eq_ci_03 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_03.csv"), data.table=FALSE)
eq_ci_04 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_04.csv"), data.table=FALSE)
eq_ci_05 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_05.csv"), data.table=FALSE)
eq_ci_06 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_06.csv"), data.table=FALSE)
eq_ci_07 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_07.csv"), data.table=FALSE)
eq_ci_08 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_08.csv"), data.table=FALSE)
eq_ci_09 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_09.csv"), data.table=FALSE)
eq_ci_10 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_10.csv"), data.table=FALSE)
eq_ci_11 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_11.csv"), data.table=FALSE)
eq_ci_12 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_12.csv"), data.table=FALSE)
eq_ci_13 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_13.csv"), data.table=FALSE)
eq_ci_14 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_14.csv"), data.table=FALSE)
eq_ci_15 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_15.csv"), data.table=FALSE)

# Load credible intervals for sensitivity analyses that sets regional inequality aversion to 0.
eq_no_reg_ci_01 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_01.csv"), data.table=FALSE)
eq_no_reg_ci_02 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_02.csv"), data.table=FALSE)
eq_no_reg_ci_03 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_03.csv"), data.table=FALSE)
eq_no_reg_ci_04 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_04.csv"), data.table=FALSE)
eq_no_reg_ci_05 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_05.csv"), data.table=FALSE)
eq_no_reg_ci_06 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_06.csv"), data.table=FALSE)
eq_no_reg_ci_07 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_07.csv"), data.table=FALSE)
eq_no_reg_ci_08 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_08.csv"), data.table=FALSE)
eq_no_reg_ci_09 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_09.csv"), data.table=FALSE)
eq_no_reg_ci_10 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_10.csv"), data.table=FALSE)
eq_no_reg_ci_11 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_11.csv"), data.table=FALSE)
eq_no_reg_ci_12 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_12.csv"), data.table=FALSE)
eq_no_reg_ci_13 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_13.csv"), data.table=FALSE)
eq_no_reg_ci_14 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_14.csv"), data.table=FALSE)
eq_no_reg_ci_15 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "ci_scch4_equity_no_regional_15.csv"), data.table=FALSE)

# Combine all regions into plot-friendly data format for full range of eta values.
fund_regions = c("usa", "canada", "western_europe", "japan_south_korea", "australia_new_zealand", "central_eastern_europe", "former_soviet_union", "middle_east", "central_america", "south_america", "south_asia", "southeast_asia", "china_plus", "north_africa", "sub_saharan_africa", "small_island_states")
equity_ci_data = list()
equity_noreg_ci_data = list()

for(r in 1:16){
	equity_ci_data[[fund_regions[r]]] = cbind(seq(0,1.5,by=0.1), rbind(eq_ci_00[r,2:4], eq_ci_01[r,2:4], eq_ci_02[r,2:4], eq_ci_03[r,2:4], eq_ci_04[r,2:4], eq_ci_05[r,2:4], eq_ci_06[r,2:4], eq_ci_07[r,2:4], eq_ci_08[r,2:4], eq_ci_09[r,2:4], eq_ci_10[r,2:4], eq_ci_11[r,2:4], eq_ci_12[r,2:4], eq_ci_13[r,2:4], eq_ci_14[r,2:4], eq_ci_15[r,2:4]))
	colnames(equity_ci_data[[fund_regions[r]]]) = c("eta", "mean", "lower_ci", "upper_ci")
}

for(r in 1:16){
	equity_noreg_ci_data[[fund_regions[r]]] = cbind(seq(0,1.5,by=0.1), rbind(eq_ci_00[r,2:4], eq_no_reg_ci_01[r,2:4], eq_no_reg_ci_02[r,2:4], eq_no_reg_ci_03[r,2:4], eq_no_reg_ci_04[r,2:4], eq_no_reg_ci_05[r,2:4], eq_no_reg_ci_06[r,2:4], eq_no_reg_ci_07[r,2:4], eq_no_reg_ci_08[r,2:4], eq_no_reg_ci_09[r,2:4], eq_no_reg_ci_10[r,2:4], eq_no_reg_ci_11[r,2:4], eq_no_reg_ci_12[r,2:4], eq_no_reg_ci_13[r,2:4], eq_no_reg_ci_14[r,2:4], eq_no_reg_ci_15[r,2:4]))
	colnames(equity_noreg_ci_data[[fund_regions[r]]]) = c("eta", "mean", "lower_ci", "upper_ci")
}

# Load individual equity-weighted SC-CH4 estimates for eta = 1.0 and a sensitivity analysis where regional inequality aversion = 0.
equity_10 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_10.csv"), data.table=FALSE)
equity_no_regional_10 = fread(file.path("results", results_folder_name, "scch4_estimates", "equity_weighting", "fund", "s_magicc", "scch4_equity_no_regional_10.csv"), data.table=FALSE)

#------------------------------------------------------
# Equity-Weighted SC-CH4 Across Different Elasticities
#------------------------------------------------------

# Set colors and indices for different regions.
usa_color                = "#5BC0EB"
western_europe_color     = "#FDE74C"
central_america_color    = "#9BC53D"
china_plus_color         = "#FAA421"
sub_saharan_africa_color = "#E55934"

plot_region_colors = c(usa_color, western_europe_color, central_america_color, china_plus_color, sub_saharan_africa_color)
plot_regions = c("usa", "western_europe", "central_america", "china_plus", "sub_saharan_africa")

plot_region_indices = match(plot_regions, colnames(equity_10))

# Set axis settings for Figure 5a.
axis_5a_settings = list(x_lim=c(0,1.5), x_breaks=seq(0,1.5,by=0.3), x_labels=as.character(seq(0,1.5,by=0.3)), x_title="Consumption Elasticity of Marginal Utility",
				        y_lim=c(-300,56000), y_breaks=c(0,10000,20000,30000,40000,50000), y_labels=c("0","10,000","20,000","30,000", "40,000", "50,000"), y_title="Equity-Weighted Social Cost \n of Methane ($/t-CH4)")

# Create Figure 5a.
fig_5a = equity_ci(equity_ci_data, plot_regions, plot_region_colors, c(0.8,0.5,0.6,0.55,0.55), axis_5a_settings, 0.7, c(5,7,1,1), FALSE, 0.0, FALSE)

# Add line segments to Figure 5a.
fig_5a = fig_5a + geom_hline(yintercept=4000, linetype="32", color="gray50", size=0.25)
fig_5a = fig_5a + geom_segment(aes(x=1.5, y=0, xend=1.5, yend=4000), linetype="32", color="gray50", size=0.25)
fig_5a = fig_5a + geom_vline(xintercept=1.0, linetype="solid", color="red", size=0.2)

# Set axis settigns for Figure 5b.
axis_5b_settings = list(x_lim=c(0,1.5), x_breaks=seq(0,1.5,by=0.3), x_labels=as.character(seq(0,1.5,by=0.3)), x_title="Consumption Elasticity of Marginal Utility",
				        y_lim=c(0,4000), y_breaks=c(0,1000,2000,3000,4000), y_labels=c("0","1,000","2,000","3,000", "4,000"), y_title="")

fig_5b = equity_ci(equity_ci_data, plot_regions, plot_region_colors, c(0.8,0.7,0.6,0.55,0.55), axis_5b_settings, 1.0, c(5,7,1,1), FALSE, 0, TRUE)

# Add dashed line for sensitivity results where regional inequality aversion = 0.
fig_5b = fig_5b + geom_line(data=equity_noreg_ci_data[[plot_regions[1]]], aes_string(x="eta", y="mean"), size=0.25, colour="black", linetype="22")


#----------------------------------------
# Equity-Weighted SC-CH4 Distributions
#----------------------------------------

# Isolate equity-weghted estimates for eta = 1.0.
equity_pdf_data = list(region1=equity_10[,plot_region_indices[5]], region2=equity_10[,plot_region_indices[4]], region3=equity_10[,plot_region_indices[3]], region4=equity_10[,plot_region_indices[2]], region5=equity_10[,plot_region_indices[1]])

# Get mean values.
mean_vals = data.frame(zeros=rep(0,6), scch4=c(mean(equity_10[,plot_region_indices[5]]), mean(equity_10[,plot_region_indices[4]]), mean(equity_no_regional_10$usa), mean(equity_10[,plot_region_indices[3]]), mean(equity_10[,plot_region_indices[2]]), mean(equity_10[,plot_region_indices[1]])))

# Set distribution colors.
pdf_region_colors = c(sub_saharan_africa_color, china_plus_color, central_america_color, western_europe_color, usa_color)
mean_colors = c(sub_saharan_africa_color, china_plus_color, "white", central_america_color, western_europe_color, usa_color)

# Set axis settings for Figure 5c.
axis_5c_settings = list(x_lim=c(-450,14000), x_breaks=seq(0,13500,by=3000), x_labels=c("0", "3,000", "6,000", "9,000", "12,000"), x_title="Equity-Weighted Social Cost of Methane ($/t-CH4)",
				        y_lim=c(0,0.0018), y_breaks=c(0,0.0005,0.001, 0.0015), y_labels=c("0","0.0005","0.001", "0.0015"), y_title="Density")

# Create Figure 5c.
fig_5c = equity_pdfs(equity_pdf_data, pdf_region_colors, rep(0.7, 5), axis_5c_settings, 0.25, c(2,7,1,4), "red", FALSE, 0)

# Add density for sensitivity without inequality aversion (intertemporal = 0.0) and all mean estimate points.
fig_5c = fig_5c + geom_density(aes(x=equity_no_regional_10$usa), colour="black", size=0.25, linetype="22")
fig_5c = fig_5c + geom_point(data=mean_vals, aes(x=scch4, y=zeros), shape=21, fill=mean_colors, size=1.4, stroke=0.3)

# Combine figures for top and bottom part of Figure 5.
fig_5_top = ggarrange(fig_5a, fig_5b, labels=c("a", "b"), nrow=1, ncol=2, label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=1.5))
fig_5_bottom = ggarrange(fig_5c, labels=c("c"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold")))

# Combine all panels.
fig_5 = grid.arrange(fig_5_top, fig_5_bottom, nrow=2, ncol=1, heights=c(2,1.2))

# Save a .jpg and .pdf version of Figure 5.
ggsave(fig_5, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Figure_5.jpg"), device="jpeg", type="cairo", width=130, height=90, unit="mm", dpi=300)
ggsave(fig_5, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Figure_5.pdf"), device="pdf", width=130, height=90, unit="mm", useDingbats = FALSE)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Extended Data Figure 1 - Additional Climate Model Hindcasts.
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Load credible interval results for baseline projections.
fair_co2_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_co2.csv"))
fund_co2_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_co2.csv"))
hector_co2_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_co2.csv"))
magicc_co2_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_co2.csv"))

fair_oceanco2_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_oceanco2_flux.csv"))
fund_oceanco2_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_oceanco2_flux.csv"))
hector_oceanco2_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_oceanco2_flux.csv"))
magicc_oceanco2_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_oceanco2_flux.csv"))

fair_oceanheat_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_ocean_heat.csv"))
fund_oceanheat_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_ocean_heat.csv"))
hector_oceanheat_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_ocean_heat.csv"))
magicc_oceanheat_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "ci_ocean_heat.csv"))


# Set some common figure settings.
mean_width = 0.3
point_size = 0.8
point_stroke = 0.2
outer_ci_width = 0.2
inner_ci_width = 0.2
plot_years = c(1850, 2022)
point_color = "yellow"
ecs_color = "gray70"

#----------------------------------
# CO2 Concentration Hindcasts.
#----------------------------------

# Create atmospheric CO2 concentration hindcasts for each for each version of SNEASY+CH4 and add in ice core observations.

fair_co2_hindcast   = climate_projection(fair_co2_ci_base, obs, "maunaloa_co2_obs", fairch4_color, c(250, 450), plot_years, "Year", expression(paste("    Carbon Dioxide \n Concentration (ppm)")), c(250, 300, 350, 400, 450), c("250", "300", "350", "400", "450"), FALSE, TRUE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FAIR", c(1856, 436), FALSE, NULL, NULL)
fair_co2_hindcast   = fair_co2_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_co2_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)
# Override margins to allow room for title
fair_co2_hindcast   = fair_co2_hindcast + theme(plot.margin = unit(c(0.4,0.1,0.1,0.5), "cm"))

fund_co2_hindcast   = climate_projection(fund_co2_ci_base, obs, "maunaloa_co2_obs", fundch4_color, c(250, 450), plot_years, "Year", expression(paste("Carbon Dioxide \n Concentration (ppm)")), c(250, 300, 350, 400, 450), c("250", "300", "350", "400", "450"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FUND", c(1856, 436), FALSE, NULL, NULL)
fund_co2_hindcast   = fund_co2_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_co2_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

hector_co2_hindcast = climate_projection(hector_co2_ci_base, obs, "maunaloa_co2_obs", hectorch4_color, c(250, 450), plot_years, "Year", expression(paste("Carbon Dioxide \n Concentration (ppm)")), c(250, 300, 350, 400, 450), c("250", "300", "350", "400", "450"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-Hector", c(1856, 436), FALSE, NULL, NULL)
hector_co2_hindcast = hector_co2_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_co2_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

magicc_co2_hindcast = climate_projection(magicc_co2_ci_base, obs, "maunaloa_co2_obs", magiccch4_color, c(250, 450), plot_years, "Year", expression(paste("Carbon Dioxide \n Concentration (ppm)")), c(250, 300, 350, 400, 450), c("250", "300", "350", "400", "450"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-MAGICC", c(1856, 436), FALSE, NULL, NULL)
magicc_co2_hindcast = magicc_co2_hindcast + geom_point(data=obs, aes_string(x="year", y="lawdome_co2_obs"), shape=24, size=point_size*0.9, stroke=point_stroke, color="black", fill=point_color)

#----------------------------------
# Ocean Carbon Flux Hindcasts.
#----------------------------------

# Create ocean carbon flux hindcasts for each version of SNEASY+CH4.

fair_oceanco2_hindcast   = climate_projection(fair_oceanco2_ci_base, obs, "oceanco2_flux_obs", fairch4_color, c(-8.5, 8.5), plot_years, "Year", expression(paste("  Ocean Carbon \n Uptake (GtC/yr)")), c(-8, -4, 0, 4, 8), c("-8", "-4", "0", "4", "8"), FALSE, TRUE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FAIR", c(1856, 7.31), FALSE, NULL, NULL)
fair_oceanco2_hindcast   = fair_oceanco2_hindcast + theme(plot.margin = unit(c(0.4,0.1,0.1,0.5), "cm"))

fund_oceanco2_hindcast   = climate_projection(fund_oceanco2_ci_base, obs, "oceanco2_flux_obs", fundch4_color, c(-8.5, 8.5), plot_years, "Year", expression(paste("  Ocean Carbon \n Uptake (GtC/yr)")), c(-8, -4, 0, 4, 8), c("-8", "-4", "0", "4", "8"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FUND", c(1856, 7.31), FALSE, NULL, NULL)

hector_oceanco2_hindcast = climate_projection(hector_oceanco2_ci_base, obs, "oceanco2_flux_obs", hectorch4_color, c(-8.5, 8.5), plot_years, "Year", expression(paste("  Ocean Carbon \n Uptake (GtC/yr)")), c(-8, -4, 0, 4, 8), c("-8", "-4", "0", "4", "8"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-Hector", c(1856, 7.31), FALSE, NULL, NULL)

magicc_oceanco2_hindcast = climate_projection(magicc_oceanco2_ci_base, obs, "oceanco2_flux_obs", magiccch4_color, c(-8.5, 8.5), plot_years, "Year", expression(paste("  Ocean Carbon \n Uptake (GtC/yr)")), c(-8, -4, 0, 4, 8), c("-8", "-4", "0", "4", "8"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-MAGICC", c(1856, 7.31), FALSE, NULL, NULL)

#----------------------------------
# Ocean Heat Content Hindcasts.
#----------------------------------

# Create ocean heat content hindcasts for each version of SNEASY+CH4.

fair_oceanheat_hindcast   = climate_projection(fair_oceanheat_ci_base, obs, "ocean_heat_obs", fairch4_color, c(-70, 70), plot_years, "Year", expression(paste("Global Ocean Heat \n Content (10^22 J)")), c(-60,0,60), c("-60", "0", "60"), TRUE, TRUE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FAIR", c(1856, 60.2), FALSE, NULL, NULL)
fair_oceanheat_hindcast   = fair_oceanheat_hindcast + theme(plot.margin = unit(c(0.4,0.1,0.1,0.5), "cm"))

fund_oceanheat_hindcast   = climate_projection(fund_oceanheat_ci_base, obs, "ocean_heat_obs", fundch4_color, c(-70, 70), plot_years, "Year", expression(paste("Global Ocean Heat \n Content (10^22 J)")), c(-60,0,60), c("-60", "0", "60"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-FUND", c(1856, 60.2), FALSE, NULL, NULL)

hector_oceanheat_hindcast = climate_projection(hector_oceanheat_ci_base, obs, "ocean_heat_obs", hectorch4_color, c(-70, 70), plot_years, "Year", expression(paste("Global Ocean Heat \n Content (10^22 J)")), c(-60,0,60), c("-60", "0", "60"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-Hector", c(1856, 60.2), FALSE, NULL, NULL)

magicc_oceanheat_hindcast = climate_projection(magicc_oceanheat_ci_base, obs, "ocean_heat_obs", magiccch4_color, c(-70, 70), plot_years, "Year", expression(paste("Global Ocean Heat \n Content (10^22 J)")), c(-60,0,60), c("-60", "0", "60"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, "solid", mean_width, "S-MAGICC", c(1856, 60.2), FALSE, NULL, NULL)

# Create some panel settings.
extended_fig_1_labels = c("a","b","c","d", "e", "f", "g", "h", "i", "j", "k", "l")
extended_fig_1_label_design = list(gp = grid::gpar(fontsize = 8, fontface="bold"))

# Arranage individual hindcasts into a single figure.
extended_fig_1 = ggarrange(fair_co2_hindcast, fund_co2_hindcast, hector_co2_hindcast, magicc_co2_hindcast,
		  		           fair_oceanco2_hindcast, fund_oceanco2_hindcast, hector_oceanco2_hindcast, magicc_oceanco2_hindcast,
   					       fair_oceanheat_hindcast, fund_oceanheat_hindcast, hector_oceanheat_hindcast, magicc_oceanheat_hindcast,
					       nrow=3, ncol=4, labels=extended_fig_1_labels, label.args = extended_fig_1_label_design)

# Save a .jpg and .pdf version of Extended Data Figure 1.
ggsave(extended_fig_1, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_1.jpg"), device="jpeg", type="cairo", width=183, height=105, unit="mm", dpi=300)
ggsave(extended_fig_1, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_1.pdf"), device="pdf", width=183, height=105, unit="mm", useDingbats = FALSE)



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# Extended Data Figure 2 - SC-CH4 Scenario PDFs For Other Models.
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# Load different scenario SC-CH4 estimates for FUND.
fund_scch4_base_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_base_fairch4) = "scch4"
fund_scch4_oldrf_fairch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_oldrf_fairch4) = "scch4"
fund_scch4_nocorr_fairch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_nocorr_fairch4) = "scch4"
fund_scch4_ecs_fairch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_ecs_fairch4) = "scch4"

fund_scch4_base_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_base_fundch4) = "scch4"
fund_scch4_oldrf_fundch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_oldrf_fundch4) = "scch4"
fund_scch4_nocorr_fundch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_nocorr_fundch4) = "scch4"
fund_scch4_ecs_fundch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_ecs_fundch4) = "scch4"

fund_scch4_base_hectorch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_base_hectorch4) = "scch4"
fund_scch4_oldrf_hectorch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_oldrf_hectorch4) = "scch4"
fund_scch4_nocorr_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_nocorr_hectorch4) = "scch4"
fund_scch4_ecs_hectorch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(fund_scch4_ecs_hectorch4) = "scch4"

# Load different scenario SC-CH4 estimates for DICE.
dice_scch4_base_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_base_fairch4) = "scch4"
dice_scch4_oldrf_fairch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_oldrf_fairch4) = "scch4"
dice_scch4_nocorr_fairch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_nocorr_fairch4) = "scch4"
dice_scch4_ecs_fairch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_ecs_fairch4) = "scch4"

dice_scch4_base_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_base_fundch4) = "scch4"
dice_scch4_oldrf_fundch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_oldrf_fundch4) = "scch4"
dice_scch4_nocorr_fundch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_nocorr_fundch4) = "scch4"
dice_scch4_ecs_fundch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_ecs_fundch4) = "scch4"

dice_scch4_base_hectorch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_base_hectorch4) = "scch4"
dice_scch4_oldrf_hectorch4  = fread(file.path("results", results_folder_name, "scch4_estimates", "outdated_ch4_forcing", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_oldrf_hectorch4) = "scch4"
dice_scch4_nocorr_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "remove_correlations", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_nocorr_hectorch4) = "scch4"
dice_scch4_ecs_hectorch4    = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE); colnames(dice_scch4_ecs_hectorch4) = "scch4"

# Set colors and transparency values for all panels.
scenario_colors = c(forcing_color, base_color, corr_color, us_ecs_color)
alphas = rep(1.0, 4)
line_size = 0.3

#----------------------------------
# SNEASY+FAIR-CH4 Distributions
#----------------------------------

# Calculate means SC-CH4 estimates for each model-scenario pair.
scenario_means_fairch4 = data.frame(zeros = c(0,0,0,0),
                                fund  = c(mean(fund_scch4_oldrf_fairch4[,1]), mean(fund_scch4_base_fairch4[,1]), mean(fund_scch4_nocorr_fairch4[,1]), mean(fund_scch4_ecs_fairch4[,1])),
                                dice  = c(mean(dice_scch4_oldrf_fairch4[,1]), mean(dice_scch4_base_fairch4[,1]), mean(dice_scch4_nocorr_fairch4[,1]), mean(dice_scch4_ecs_fairch4[,1])))

# Plot SC-CH4 distributions for FUND + FAIR-CH4.
extended_fig_2a_top = scch4_pdf_scenario(fund_scch4_oldrf_fairch4, fund_scch4_base_fairch4, fund_scch4_nocorr_fairch4, fund_scch4_ecs_fairch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "", FALSE, c(0,0.003), "solid", c(0.5,0.4,0.2,0))
extended_fig_2a_top = extended_fig_2a_top + geom_point(data=scenario_means_fairch4, aes_string(x="fund", y="zeros"), shape=21, size=1.75, fill=scenario_colors, stroke=0.2)

# Plot SC-CH4 distributions for DICE + FAIR-CH4.
extended_fig_2a_bottom = scch4_pdf_scenario(dice_scch4_oldrf_fairch4, dice_scch4_base_fairch4, dice_scch4_nocorr_fairch4, dice_scch4_ecs_fairch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "Social Cost of Methane ($/t-CH4)", TRUE, c(0,0.0033), "22", c(-3.2,0.4,0.2,0))
extended_fig_2a_bottom = extended_fig_2a_bottom + geom_point(data=scenario_means_fairch4, aes_string(x="dice", y="zeros"), shape=23, size=1.75, fill=scenario_colors, stroke=0.2)

#----------------------------------
# SNEASY+FUND-CH4 Distributions
#----------------------------------

# Calculate means SC-CH4 estimates for each model-scenario pair.
scenario_means_fundch4 = data.frame(zeros = c(0,0,0,0),
                                fund  = c(mean(fund_scch4_oldrf_fundch4[,1]), mean(fund_scch4_base_fundch4[,1]), mean(fund_scch4_nocorr_fundch4[,1]), mean(fund_scch4_ecs_fundch4[,1])),
                                dice  = c(mean(dice_scch4_oldrf_fundch4[,1]), mean(dice_scch4_base_fundch4[,1]), mean(dice_scch4_nocorr_fundch4[,1]), mean(dice_scch4_ecs_fundch4[,1])))

# Plot SC-CH4 distributions for FUND + FUND-CH4.
extended_fig_2b_top = scch4_pdf_scenario(fund_scch4_oldrf_fundch4, fund_scch4_base_fundch4, fund_scch4_nocorr_fundch4, fund_scch4_ecs_fundch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "", FALSE, c(0,0.003), "solid", c(0.5,0.3,0.2,0.1))
extended_fig_2b_top = extended_fig_2b_top + geom_point(data=scenario_means_fundch4, aes_string(x="fund", y="zeros"), shape=21, size=1.75, fill=scenario_colors, stroke=0.2)

# Plot SC-CH4 distributions for DICE + FUND-CH4.
extended_fig_2b_bottom = scch4_pdf_scenario(dice_scch4_oldrf_fundch4, dice_scch4_base_fundch4, dice_scch4_nocorr_fundch4, dice_scch4_ecs_fundch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "Social Cost of Methane ($/t-CH4)", TRUE, c(0,0.0033), "22", c(-3.2,0.3,0.2,0.1))
extended_fig_2b_bottom = extended_fig_2b_bottom + geom_point(data=scenario_means_fundch4, aes_string(x="dice", y="zeros"), shape=23, size=1.75, fill=scenario_colors, stroke=0.2)

#----------------------------------
# SNEASY+MAGICC-CH4 Distributions
#----------------------------------

# Calculate means SC-CH4 estimates for each model-scenario pair.
scenario_means_hectorch4 = data.frame(zeros = c(0,0,0,0),
                                fund  = c(mean(fund_scch4_oldrf_hectorch4[,1]), mean(fund_scch4_base_hectorch4[,1]), mean(fund_scch4_nocorr_hectorch4[,1]), mean(fund_scch4_ecs_hectorch4[,1])),
                                dice  = c(mean(dice_scch4_oldrf_hectorch4[,1]), mean(dice_scch4_base_hectorch4[,1]), mean(dice_scch4_nocorr_hectorch4[,1]), mean(dice_scch4_ecs_hectorch4[,1])))

# Plot SC-CH4 distributions for FUND + Hector-CH4.
extended_fig_2c_top = scch4_pdf_scenario(fund_scch4_oldrf_hectorch4, fund_scch4_base_hectorch4, fund_scch4_nocorr_hectorch4, fund_scch4_ecs_hectorch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "", FALSE, c(0,0.003), "solid", c(0.5,0.2,0.2,0.2))
extended_fig_2c_top = extended_fig_2c_top + geom_point(data=scenario_means_hectorch4, aes_string(x="fund", y="zeros"), shape=21, size=1.75, fill=scenario_colors, stroke=0.2)

# Plot SC-CH4 distributions for DICE + Hector-CH4.
extended_fig_2c_bottom = scch4_pdf_scenario(dice_scch4_oldrf_hectorch4, dice_scch4_base_hectorch4, dice_scch4_nocorr_hectorch4, dice_scch4_ecs_hectorch4, alphas, scenario_colors, line_size, c(-200,5200), c(0,1000,2000,3000,4000,5000), c("0","1000","2000","3000","4000","5000"), "Social Cost of Methane ($/t-CH4)", TRUE, c(0,0.0033), "22", c(-3.2,0.2,0.2,0.2))
extended_fig_2c_bottom = extended_fig_2c_bottom + geom_point(data=scenario_means_hectorch4, aes_string(x="dice", y="zeros"), shape=23, size=1.75, fill=scenario_colors, stroke=0.2)

# Create pannel design settings.
extended_fig_2_label_deisgn = list(gp = grid::gpar(fontsize = 8, fontface="bold"))#, hjust=0.0, vjust=1.5)

# Combine DICE and FUND distributions for each version of SNEASY-CH4.
extended_fig_2a = ggarrange(extended_fig_2a_top, extended_fig_2a_bottom, ncol=1, labels=c("a", ""), label.args = extended_fig_2_label_deisgn)
extended_fig_2b = ggarrange(extended_fig_2b_top, extended_fig_2b_bottom, ncol=1, labels=c("b", ""), label.args = extended_fig_2_label_deisgn)
extended_fig_2c = ggarrange(extended_fig_2c_top, extended_fig_2c_bottom, ncol=1, labels=c("c", ""), label.args = extended_fig_2_label_deisgn)

# Combine all panels into Extended Data Figure 2.
extended_fig_2 = grid.arrange(extended_fig_2a, extended_fig_2b, extended_fig_2c, widths=c(1,1,1), nrow=1, ncol=3)

# Save a .jpg and .pdf version of Extended Data Figure 2.
ggsave(extended_fig_2, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_2.jpg"), device="jpeg", type="cairo", width=180, height=95, unit="mm", dpi=300)
ggsave(extended_fig_2, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_2.pdf"), device="pdf", width=180, height=95, unit="mm", useDingbats = FALSE)



#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Extended Data Figure 3 - RCP2.6 Temperature Hindcasts and SC-CH4 Distributions.
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

# Load RCP 2.6 temperature projection credible intervals.
fair_rcp26_ci_temperature   = read.csv(file.path("results", results_folder_name, "climate_projections", "rcp26", "s_fair", "ci_temperature.csv"))
fund_rcp26_ci_temperature   = read.csv(file.path("results", results_folder_name, "climate_projections", "rcp26", "s_fund", "ci_temperature.csv"))
hector_rcp26_ci_temperature = read.csv(file.path("results", results_folder_name, "climate_projections", "rcp26", "s_hector", "ci_temperature.csv"))
magicc_rcp26_ci_temperature = read.csv(file.path("results", results_folder_name, "climate_projections", "rcp26", "s_magicc", "ci_temperature.csv"))

# Load RCP 2.6 SC-CH4 results for DICE.
dice_scch4_fairch4_rcp26   = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)
dice_scch4_fundch4_rcp26   = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)
dice_scch4_hectorch4_rcp26 = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)
dice_scch4_magiccch4_rcp26 = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load RCP 2.6 SC-CH4 results for FUND.
fund_scch4_fairch4_rcp26   = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)
fund_scch4_fundch4_rcp26   = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)
fund_scch4_hectorch4_rcp26 = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)
fund_scch4_magiccch4_rcp26 = fread(file.path("results", results_folder_name, "scch4_estimates", "rcp26", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)

#----------------------------------
# RCP 2.6 Temperature Projections
#----------------------------------

# Set some common figure settings.
mean_width = 0.3
point_size = 1.25
point_stroke = 0.2
outer_ci_width = 0.25
inner_ci_width = 0.2

# Make RCP 2.6 tempertaure projections.
extended_fig_3a = rcp26_projection(fair_rcp26_ci_temperature, obs, "hadcrut_temperature_obs", fairch4_color, 0.8, c(-1.0, 2.5), c(1850,2300), "Year", expression(paste("Surface Temperature ("~degree*C*")")), c(-1.0, 0.0, 1.0, 2.0), c("-1", "0", "1", "2"), FALSE, TRUE, point_size, point_stroke, outer_ci_width, mean_width, "S-FAIR", c(1860, 2.35), c(0.25,0.2,0.2,0.3))
extended_fig_3b = rcp26_projection(fund_rcp26_ci_temperature, obs, "hadcrut_temperature_obs", fundch4_color, 0.8, c(-1.0, 2.5), c(1850,2300), "Year", expression(paste("Surface Temperature ("~degree*C*")")), c(-1.0, 0.0, 1.0, 2.0), c("-1", "0", "1", "2"), FALSE, FALSE, point_size, point_stroke, outer_ci_width, mean_width, "S-FUND", c(1860, 2.35), c(0.25,0.2,0.2,0.2))
extended_fig_3c = rcp26_projection(hector_rcp26_ci_temperature, obs, "hadcrut_temperature_obs", hectorch4_color, 0.8, c(-1.0, 2.5), c(1850,2300), "Year", expression(paste("Surface Temperature ("~degree*C*")")), c(-1.0, 0.0, 1.0, 2.0), c("-1", "0", "1", "2"), TRUE, TRUE, point_size, point_stroke, outer_ci_width, mean_width, "S-Hector", c(1860, 2.35),c(0.25,0.2,0.2,0.3))
extended_fig_3d = rcp26_projection(magicc_rcp26_ci_temperature, obs, "hadcrut_temperature_obs", magiccch4_color, 0.8, c(-1.0, 2.5), c(1850,2300), "Year", expression(paste("Surface Temperature ("~degree*C*")")), c(-1.0, 0.0, 1.0, 2.0), c("-1", "0", "1", "2"), TRUE, FALSE, point_size, point_stroke, outer_ci_width, mean_width, "S-MAGICC", c(1860, 2.35), c(0.25,0.2,0.2,0.2))

#----------------------------------
# RCP 2.6 SC-CH4 Distributions
#----------------------------------

# Set colors and transparencies for different versions of SNEASY+CH4.
scch4_colors = c(fundch4_color, hectorch4_color, fairch4_color, magiccch4_color)
alphas = rep(1.0, 4)

# Merge baseline data into a data.frame for plotting.
scch4_rcp26_fairch4   = data.frame(fund=fund_scch4_fairch4_rcp26[,1], dice=dice_scch4_fairch4_rcp26[,1])
scch4_rcp26_fundch4   = data.frame(fund=fund_scch4_fundch4_rcp26[,1], dice=dice_scch4_fundch4_rcp26[,1])
scch4_rcp26_hectorch4 = data.frame(fund=fund_scch4_hectorch4_rcp26[,1], dice=dice_scch4_hectorch4_rcp26[,1])
scch4_rcp26_magiccch4 = data.frame(fund=fund_scch4_magiccch4_rcp26[,1], dice=dice_scch4_magiccch4_rcp26[,1])


# Calculate IAM mean SC-CH4 estimates.
rcp26_dice_mean = mean(c(dice_scch4_fairch4_rcp26[,1], dice_scch4_fundch4_rcp26[,1], dice_scch4_hectorch4_rcp26[,1], dice_scch4_magiccch4_rcp26[,1]))
rcp26_fund_mean = mean(c(fund_scch4_fairch4_rcp26[,1], fund_scch4_fundch4_rcp26[,1], fund_scch4_hectorch4_rcp26[,1], fund_scch4_magiccch4_rcp26[,1]))

# Combine with baseline SC-CH4 means from Figure 2.
rcp_mean_data = data.frame(zeros=c(0,0,0,0), means=c(rcp26_fund_mean, base_fund_mean, rcp26_dice_mean, base_dice_mean))

# Create RCP 2.6 SC-CH4 distributions and add points for mean estimates.
extended_fig_3e = scch4_pdf_baseline(scch4_rcp26_fundch4, scch4_rcp26_hectorch4, scch4_rcp26_fairch4, scch4_rcp26_magiccch4, alphas, scch4_colors, 0.35, c(-100,2100), c(0,500,1000,1500,2000), c("0","500","1000","1500","2000"), "Social Cost of Methane ($/t-CH4)", c(-0.00035,0.003))
extended_fig_3e = extended_fig_3e + geom_point(data=rcp_mean_data, aes_string(x="means", y="zeros"),   shape=c(21,21,23,23), size=2.75,  fill="white", stroke=0.5)

# Merge temperature projections into a single panel.
extended_fig_3_top = ggarrange(extended_fig_3a, extended_fig_3b, extended_fig_3c, extended_fig_3d, nrow=2, ncol=2, labels=c("a", "b", "c", "d"), label.args = list(gp = grid::gpar(fontsize = 8, fontface="bold")))

# Turn distribution plot into a labeled panel.
extended_fig_3_bottom = ggarrange(extended_fig_3e, labels=c("e"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=0.5))

# Combine everything into Extended Data Figure 3.
extended_fig_3 = grid.arrange(extended_fig_3_top, extended_fig_3_bottom, heights=c(2,1.3), nrow=2, ncol=1)

# Save a .jpg and .pdf version of Extended Data Figure 3.
ggsave(extended_fig_3, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_3.jpg"), device="jpeg", type="cairo", width=130, height=160, unit="mm", dpi=600)
ggsave(extended_fig_3, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_3.pdf"), device="pdf", width=130, height=160, unit="mm", useDingbats = FALSE)


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Extended Data Figure 4 - Wider Prior SC-CH4 Distributions and Methane Cycle Correlations
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------

# Load baseline SC-CH4 results for DICE.
dice_scch4_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)
dice_scch4_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)
dice_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)
dice_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load baseline SC-CH4 results for FUND.
fund_scch4_fairch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)
fund_scch4_fundch4   = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)
fund_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)
fund_scch4_magiccch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load SC-CH4 results for DICE using wider prior parameter distributions.
dice_scch4_fairch4_wider   = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)
dice_scch4_fundch4_wider   = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)
dice_scch4_hectorch4_wider = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)
dice_scch4_magiccch4_wider = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Load SC-CH4 results for FUND using wider prior parameter distributions.
fund_scch4_fairch4_wider   = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)
fund_scch4_fundch4_wider   = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)
fund_scch4_hectorch4_wider = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)
fund_scch4_magiccch4_wider = fread(file.path("results", results_folder_name, "scch4_estimates", "wider_priors", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)

# Pool baseline and wider prior SC-CH4 results by IAM.
baseline_dice = data.frame(scch4=c(dice_scch4_fairch4[,1], dice_scch4_fundch4[,1], dice_scch4_hectorch4[,1], dice_scch4_magiccch4[,1]))
baseline_fund = data.frame(scch4=c(fund_scch4_fairch4[,1], fund_scch4_fundch4[,1], fund_scch4_hectorch4[,1], fund_scch4_magiccch4[,1]))

wider_dice = data.frame(scch4=c(dice_scch4_fairch4_wider[,1], dice_scch4_fundch4_wider[,1], dice_scch4_hectorch4_wider[,1], dice_scch4_magiccch4_wider[,1]))
wider_fund = data.frame(scch4=c(fund_scch4_fairch4_wider[,1], fund_scch4_fundch4_wider[,1], fund_scch4_hectorch4_wider[,1], fund_scch4_magiccch4_wider[,1]))

#Load baseline posterior parameters for scatter plot.
posterior_param_magiccch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_magicc", "parameters_100k.csv"), data.table=FALSE)

#----------------------------------
# Wider Prior SC-CH4 Distribution
#----------------------------------

# Calculate mean SC-CH4 estimates for baseline and wider prior scenarios.
wider_mean = mean(c(wider_dice[,1], wider_fund[,1]))
base_mean  = mean(c(baseline_dice[,1], baseline_fund[,1]))

# Create distribution plot and add points multi-model mean estimates.
extended_fig_4a = scch4_wider_pdfs(baseline_dice, baseline_fund, wider_dice, wider_fund, "red", "dodgerblue", 0.5, 0.8, c(-200,3600), c(0,1000,2000,3000,4000), c("0","1,000","2,000","3,000","4,000"), "Social Cost of Methane ($/t-CH4)", c(0,0.0023))
extended_fig_4a = extended_fig_4a + geom_point(aes(x=c(wider_mean,base_mean), y=c(0,0)), shape=21, color="black", stroke=0.5, size=2.5, fill=c("dodgerblue", "red"))

#---------------------------------------
# Methane Cycle Parameter Scatter Plots
#---------------------------------------

# Select methane cycle parameters to plot.
scatter_data_ch4cycle  = data.frame(lifetime=post_param_magiccch4$tau_0, natural_emiss=post_param_magiccch4$CH4_nat, dice=dice_scch4_magiccch4[,1], fund=fund_scch4_magiccch4[,1])

# Create scatter plot between intitial CH4 tropospheric lifetime and natural CH4 emission rates.
extended_fig_4b = ch4cycle_scatter(scatter_data_ch4cycle, c("natural_emiss", "lifetime"), 5000, 21, 1.0, "mediumorchid1", 0.8, 1,1, "1", c(175,320), c(180, 220, 260, 300), c("180", "220", "260", "300"), c(6.6,9.1), c(7,8,9), c("7","8","9"), "Natural methane emissions (Mt/yr)", "Initial tropospheric methane lifetime (years)", "N/A", c(6,2,3,3), TRUE, 2, FALSE, "", "", "")

# Create scatter plot between methane cycle parameters and SC-CH4.
extended_fig_4c = ch4cycle_scatter(scatter_data_ch4cycle, c("natural_emiss", "dice", "lifetime"), 5000, 23, 1.0, "dodgerblue1", "", c(0.2,3.8), c(7,7.5,8,8.5), c("7", "7.5", "8", "8.5"), c(175,320), c(180, 220, 260, 300), c("180", "220", "260", "300"), c(0,3600), c(0,1000,2000,3000), c("0","1,000","2,000","3,000"), "Natural methane emissions (Mt/yr)", "Social cost of methane ($/t-CH4)", c("Initial tropospheric \nmethane lifetime (years)"), c(6,2,3,3), FALSE, "", TRUE, c("natural_emiss", "fund", "lifetime"), 21, "indianred1")

# Create panels and combine into Extended Data Figure 4.
extended_fig_4_top    = ggarrange(extended_fig_4a, labels=c("a"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold", vjust=-3)))
extended_fig_4_bottom = ggarrange(extended_fig_4b, extended_fig_4c, labels=c("b", "c"), nrow=1, ncol=2, label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold")))
extended_fig_4        = grid.arrange(extended_fig_4_top, extended_fig_4_bottom, nrow=2, ncol=1, heights=c(1,0.8))

# Save a .jpg and .pdf version of Extended Data Figure 4.
ggsave(extended_fig_4, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_4.jpg"), device="jpeg", type="cairo", width=136, height=160, unit="mm", dpi=300)
ggsave(extended_fig_4, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_4.pdf"), device="pdf", width=136, height=160, unit="mm", useDingbats = FALSE)


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Extended Data Figure 5 - Parameter Scatterplots for Other Models.
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# Load climate model indices that successfully ran.
fairch4_climate_indices   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "good_indices.csv"))[,1]
fundch4_climate_indices   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "good_indices.csv"))[,1]
hectorch4_climate_indices = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "good_indices.csv"))[,1]
magiccch4_climate_indices = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_magicc", "good_indices.csv"))[,1]

# Load SC-CH4 model indices that successfully ran.
fairch4_scch4_indices_dice   = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fair", "good_indices.csv"))[,1]
fundch4_scch4_indices_dice   = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fund", "good_indices.csv"))[,1]
hectorch4_scch4_indices_dice = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_hector", "good_indices.csv"))[,1]
magiccch4_scch4_indices_dice = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_magicc", "good_indices.csv"))[,1]

fairch4_scch4_indices_fund   = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fair", "good_indices.csv"))[,1]
fundch4_scch4_indices_fund   = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fund", "good_indices.csv"))[,1]
hectorch4_scch4_indices_fund = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_hector", "good_indices.csv"))[,1]
magiccch4_scch4_indices_fund = read.csv(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_magicc", "good_indices.csv"))[,1]

# Combine indices (all successful climate runs that also ran for both DICE and FUND).
fairch4_indices   = fairch4_climate_indices[unique(c(fairch4_scch4_indices_dice, fairch4_scch4_indices_fund))]
fundch4_indices   = fundch4_climate_indices[unique(c(fundch4_scch4_indices_dice, fundch4_scch4_indices_fund))]
hectorch4_indices = hectorch4_climate_indices[unique(c(hectorch4_scch4_indices_dice, hectorch4_scch4_indices_fund))]
magiccch4_indices = magiccch4_climate_indices[unique(c(magiccch4_scch4_indices_dice, magiccch4_scch4_indices_fund))]

# Load S-FAIR posterior parameters and corresponding SC-CH4 estimates for scatter plots.
post_param_fairch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_fair", "parameters_100k.csv"), data.table=FALSE)[fairch4_indices, ]
dice_scch4_fairch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)[fairch4_indices, ]
fund_scch4_fairch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)[fairch4_indices, ]

# Load S-FUND posterior parameters and corresponding SC-CH4 estimates for scatter plots.
post_param_fundch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_fund", "parameters_100k.csv"), data.table=FALSE)[fundch4_indices, ]
dice_scch4_fundch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)[fundch4_indices, ]
fund_scch4_fundch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)[fundch4_indices, ]

# Load S-Hector posterior parameters and corresponding SC-CH4 estimates for scatter plots.
post_param_hectorch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_hector", "parameters_100k.csv"), data.table=FALSE)[hectorch4_indices, ]
dice_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)[hectorch4_indices, ]
fund_scch4_hectorch4 = fread(file.path("results", results_folder_name, "scch4_estimates", "baseline_run", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)[hectorch4_indices, ]

# Get upper bound point size value from Figure 3.
post_param_magiccch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_magicc", "parameters_100k.csv"), data.table=FALSE)[magiccch4_indices, ]
upper_bound_size = ceiling(as.numeric(quantile(post_param_magiccch4$Q10, 0.99)))

# Set Figure 3's upper bound point size value across all other models.
post_param_fairch4$Q10_size = post_param_fairch4$Q10
post_param_fairch4[which(post_param_fairch4$Q10 > upper_bound_size), "Q10_size"] = upper_bound_size

post_param_fundch4$Q10_size = post_param_fundch4$Q10
post_param_fundch4[which(post_param_fundch4$Q10 > upper_bound_size), "Q10_size"] = upper_bound_size

post_param_hectorch4$Q10_size = post_param_hectorch4$Q10
post_param_hectorch4[which(post_param_hectorch4$Q10 > upper_bound_size), "Q10_size"] = upper_bound_size

# Create data.frames of each model's parameters and SC-CH4 values for plotting.
scatter_data_fairch4   = data.frame(ECS=post_param_fairch4$ECS, aerosol=post_param_fairch4$alpha, Q10_size = post_param_fairch4$Q10_size, heat_diffusion=post_param_fairch4$kappa, dice=dice_scch4_fairch4, fund=fund_scch4_fairch4)
scatter_data_fundch4   = data.frame(ECS=post_param_fundch4$ECS, aerosol=post_param_fundch4$alpha, Q10_size = post_param_fundch4$Q10_size, heat_diffusion=post_param_fundch4$kappa, dice=dice_scch4_fundch4, fund=fund_scch4_fundch4)
scatter_data_hectorch4 = data.frame(ECS=post_param_hectorch4$ECS, aerosol=post_param_hectorch4$alpha, Q10_size = post_param_hectorch4$Q10_size, heat_diffusion=post_param_hectorch4$kappa, dice=dice_scch4_hectorch4, fund=fund_scch4_hectorch4)

#----------------------------------
# SNEASY+FAIR-CH4 Scatter Plots
#----------------------------------

# Create scatter plots for S-FAIR.
extended_fig_5a = scatter_4way(scatter_data_fairch4, c("aerosol", "ECS", "Q10_size", "heat_diffusion"), 5000, 21, 0.8, c("blue", "dodgerblue", "cyan", "yellow"), c(2.0, 4.0), c(2.0, 4.0), c("< 2.0", "> 4.0"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,2.03), c(0,0.5,1,1.5,2),  c("0","0.5","1","1.5","2"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), "Aerosol radiative forcing scale factor", "Equilibirium climate sensitivity (C)", c("Ocean heat \ndiffusivity", "Carbon sink respiration \ntemperature sensitivity"), c(6,2,3,3), FALSE, "", "", FALSE, TRUE)
extended_fig_5d = scatter_4way(scatter_data_fairch4, c("ECS", "dice", "Q10_size", "aerosol"), 5000, 23, 0.9, c("yellow", "red", "blue"), c(0.6, 1.5), c(0.6,1.5), c("< 0.5", "> 1.5"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), c(0,3600), c(0,1000,2000,3000), c("0","1,000","2,000","3,000"), "Equilibirium climate sensitivity (C)", "Social cost of methane ($/t-CH4)", c("Aerosol forcing \nscale factor", "Carbon sink respiration \ntemperature sensitivity"), c(6,3,3,2), TRUE, c("ECS", "fund", "Q10_size", "aerosol"), 21, FALSE, TRUE)

#----------------------------------
# SNEASY+FUND-CH4 Scatter Plots
#----------------------------------

# Create scatter plots for S-FUND.
extended_fig_5b = scatter_4way(scatter_data_fundch4, c("aerosol", "ECS", "Q10_size", "heat_diffusion"), 5000, 21, 0.8, c("blue", "dodgerblue", "cyan", "yellow"), c(2.0, 4.0), c(2.0, 4.0), c("< 2.0", "> 4.0"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,2.03), c(0,0.5,1,1.5,2),  c("0","0.5","1","1.5","2"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), "Aerosol radiative forcing scale factor", "Equilibirium climate sensitivity (C)", c("Ocean heat \ndiffusivity", "Carbon sink respiration \ntemperature sensitivity"), c(6,2,3,3), FALSE, "", "", FALSE, TRUE)
extended_fig_5e = scatter_4way(scatter_data_fundch4, c("ECS", "dice", "Q10_size", "aerosol"), 5000, 23, 0.9, c("yellow", "red", "blue"),c(0.6, 1.5), c(0.6,1.5), c("< 0.5", "> 1.5"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), c(0,3600), c(0,1000,2000,3000), c("0","1,000","2,000","3,000"), "Equilibirium climate sensitivity (C)", "Social cost of methane ($/t-CH4)", c("Aerosol forcing \nscale factor", "Carbon sink respiration \ntemperature sensitivity"), c(6,3,3,2), TRUE, c("ECS", "fund", "Q10_size", "aerosol"), 21, FALSE, TRUE)

#----------------------------------
# SNEASY+Hector-CH4 Scatter Plots
#----------------------------------

# Create scatter plots for S-Hector.
extended_fig_5c = scatter_4way(scatter_data_hectorch4, c("aerosol", "ECS", "Q10_size", "heat_diffusion"), 5000, 21, 0.8, c("blue", "dodgerblue", "cyan", "yellow"), c(2.0, 4.0), c(2.0, 4.0), c("< 2.0", "> 4.0"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,2.03), c(0,0.5,1,1.5,2),  c("0","0.5","1","1.5","2"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), "Aerosol radiative forcing scale factor", "Equilibirium climate sensitivity (C)", c("Ocean heat \ndiffusivity", "Carbon sink respiration \ntemperature sensitivity"), c(6,2,3,3), FALSE, "", "", TRUE, TRUE)
extended_fig_5f = scatter_4way(scatter_data_hectorch4, c("ECS", "dice", "Q10_size", "aerosol"), 5000, 23, 0.9, c("yellow", "red", "blue"), c(0.6, 1.5), c(0.6,1.5), c("< 0.5", "> 1.5"), c(0.5,4.0), c(1.01,2.0,3, 4), c("1", "2", "3", "4+"), c(0,9.5), c(0,2,4,6,8), c("0","2","4","6","8"), c(0,3600), c(0,1000,2000,3000), c("0","1,000","2,000","3,000"), "Equilibirium climate sensitivity (C)", "Social cost of methane ($/t-CH4)", c("Aerosol forcing \nscale factor", "Carbon sink respiration \ntemperature sensitivity"), c(6,3,3,2), TRUE, c("ECS", "fund", "Q10_size", "aerosol"), 21, TRUE, TRUE)

# Merge all scatter plots into a single panel.
extended_fig_5 = ggarrange(extended_fig_5a, extended_fig_5d, extended_fig_5b, extended_fig_5e, extended_fig_5c, extended_fig_5f, nrow=3, ncol=2, labels=c("a","d","b","e","c","f"), label.args = list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=2))

# Save a .jpg and .pdf version of Extended Data Figure 5.
ggsave(extended_fig_5, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_5.jpg"), device="jpeg", type="cairo", width=136, height=190, unit="mm", dpi=300)
ggsave(extended_fig_5, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_5.pdf"), device="pdf", width=136, height=190, unit="mm", useDingbats = FALSE)



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Extended Data Figure 6 - Temperature Projections Without Posterior Correlations.
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# Load individual baseline temperature projections.
fair_temperature_base   = fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "base_temperature.csv"), data.table=FALSE)
fund_temperature_base   = fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "base_temperature.csv"), data.table=FALSE)
hector_temperature_base = fread(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "base_temperature.csv"), data.table=FALSE)

# Load baseline temperature projection credible intervals.
fair_temperature_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fair", "ci_temperature.csv"))
fund_temperature_ci_base   = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_fund", "ci_temperature.csv"))
hector_temperature_ci_base = read.csv(file.path("results", results_folder_name, "climate_projections", "baseline_run", "s_hector", "ci_temperature.csv"))

# Load individual temperature projections for scenario without posterior correlations.
fair_temperature_no_corr   = fread(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_fair", "base_temperature.csv"), data.table=FALSE)
fund_temperature_no_corr   = fread(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_fund", "base_temperature.csv"), data.table=FALSE)
hector_temperature_no_corr = fread(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_hector", "base_temperature.csv"), data.table=FALSE)

# Load temperature projection credible intervals for scenario without posterior correlations.
fair_temperature_ci_no_corr   = read.csv(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_fair", "ci_temperature.csv"))
fund_temperature_ci_no_corr   = read.csv(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_fund", "ci_temperature.csv"))
hector_temperature_ci_no_corr = read.csv(file.path("results", results_folder_name, "climate_projections", "remove_correlations", "s_hector", "ci_temperature.csv"))

# Set years for pdf temperature plots.
sneasy_years = 1765:2300
pdf_years = c(2050, 2100)

# Isolate individual temperature projections for specific years.
fair_pdf_data_base = fair_temperature_base[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(fair_pdf_data_base) = c("Year_1", "Year_2")
fair_pdf_data_no_corr = fair_temperature_no_corr[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(fair_pdf_data_no_corr) = c("Year_1", "Year_2")

fund_pdf_data_base = fund_temperature_base[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(fund_pdf_data_base) = c("Year_1", "Year_2")
fund_pdf_data_no_corr = fund_temperature_no_corr[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(fund_pdf_data_no_corr) = c("Year_1", "Year_2")

hector_pdf_data_base = hector_temperature_base[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(hector_pdf_data_base) = c("Year_1", "Year_2")
hector_pdf_data_no_corr = hector_temperature_no_corr[ , which(sneasy_years==pdf_years[1] | sneasy_years==pdf_years[2])]; colnames(hector_pdf_data_no_corr) = c("Year_1", "Year_2")

# Set some plot settings.
mean_width = 0.4
ci_width = 0.25
point_size = 1.0
point_stroke = 0.2
plot_years = c(1850, 2210)
point_color = "yellow"
no_corr_color = "gray75"

#-------------------------------------------
# SNEASY+FAIR-CH4 Temperature Projections
#-------------------------------------------

# Plot S-FAIR temperature projections (with and without posterior correlations).
extended_fig_6a_main = climate_projection(fair_temperature_ci_base, obs, "hadcrut_temperature_obs", fairch4_color, c(-0.5,12.5), plot_years, "Year", "Surface Temperature Increase (*C)", seq(0,12,by=2), as.character(seq(0,12,by=2)), FALSE, TRUE, point_size, point_stroke, ci_width, "dashed", mean_width, "S-FAIR", c(1856, 11.75), TRUE, fair_temperature_ci_no_corr, no_corr_color)

# Create temperature distribution inset for S-FAIR and merge into main panel.
extended_fig_6a_inset = ggarrange(inset_nocorr_pdfs(fair_pdf_data_base, fair_pdf_data_no_corr, c(fairch4_color, no_corr_color), c(0.7,0.6), 0.2, c(0,8), c(0,2,4,6,8), c("0","2","4","6","8"), "Surface Temperature Increase (*C)", c(0,1.69), c(0,0,0,0)), nrow=1, ncol=1)
extended_fig_6a = extended_fig_6a_main + annotation_custom(grob=extended_fig_6a_inset, xmin=1860, xmax=2045, ymin=3.5, ymax=11.5)

#-------------------------------------------
# SNEASY+FUND-CH4 Temperature Projections
#-------------------------------------------

# Plot S-FUND temperature projections (with and without posterior correlations).
extended_fig_6b_main = climate_projection(fund_temperature_ci_base, obs, "hadcrut_temperature_obs", fundch4_color, c(-0.5,12.5), plot_years, "Year", "Surface Temperature Increase (*C)", seq(0,12,by=2), as.character(seq(0,12,by=2)), FALSE, TRUE, point_size, point_stroke, ci_width, "dashed", mean_width, "S-FUND", c(1856, 11.75), TRUE, fund_temperature_ci_no_corr, no_corr_color)

# Create temperature distribution inset for S-FUND and merge into main panel.
extended_fig_6b_inset = ggarrange(inset_nocorr_pdfs(fund_pdf_data_base, fund_pdf_data_no_corr, c(fundch4_color, no_corr_color), c(0.7,0.6), 0.2, c(0,8), c(0,2,4,6,8), c("0","2","4","6","8"), "Surface Temperature Increase (*C)", c(0,1.69), c(0,0,0,0)), nrow=1, ncol=1)
extended_fig_6b = extended_fig_6b_main + annotation_custom(grob=extended_fig_6b_inset, xmin=1860, xmax=2045, ymin=3.5, ymax=11.5)

#-------------------------------------------
# SNEASY+Hector-CH4 Temperature Projections
#-------------------------------------------

# Plot S-Hector temperature projections (with and without posterior correlations).
extended_fig_6c_main =  climate_projection(hector_temperature_ci_base, obs, "hadcrut_temperature_obs", hectorch4_color, c(-0.5,12.5), plot_years, "Year", "Surface Temperature Increase (*C)", seq(0,12,by=2), as.character(seq(0,12,by=2)), TRUE, TRUE, point_size, point_stroke, ci_width, "dashed", mean_width, "S-Hector", c(1856, 11.75), TRUE, hector_temperature_ci_no_corr, no_corr_color)

# Create temperature distribution inset for S-Hector and merge into main panel.
extended_fig_6c_inset = ggarrange(inset_nocorr_pdfs(hector_pdf_data_base, hector_pdf_data_no_corr, c(hectorch4_color, no_corr_color), c(0.7,0.6), 0.2, c(0,8), c(0,2,4,6,8), c("0","2","4","6","8"), "Surface Temperature Increase (*C)", c(0,1.69), c(0,0,0,0)), nrow=1, ncol=1)
extended_fig_6c = extended_fig_6c_main + annotation_custom(grob=extended_fig_6c_inset, xmin=1860, xmax=2045, ymin=3.5, ymax=11.5)

# Merge all plots into a single panel.
extended_fig_6 = ggarrange(extended_fig_6a, extended_fig_6b, extended_fig_6c, nrow=3, ncol=1, labels=c("a", "b", "c"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold")))

# Save a .jpg and .pdf version of Extended Data Figure 6.
ggsave(extended_fig_6, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_6.jpg"), device="jpeg", type="cairo", width=130, height=170, unit="mm", dpi=300)
ggsave(extended_fig_6, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_6.pdf"), device="pdf", width=130, height=170, unit="mm", useDingbats = FALSE)



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Extended Data Figure 7 - ECS Distributions and SC-CH4 vs. EPA ECS Correlation.
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# Load DICE SC-CH4 estimates that sample the U.S. climate sensivitiy distribution.
dice_scch4_fairch4_us   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "scch4_30.csv"), data.table=FALSE)[,1]
dice_scch4_fundch4_us   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "scch4_30.csv"), data.table=FALSE)[,1]
dice_scch4_hectorch4_us = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "scch4_30.csv"), data.table=FALSE)[,1]
dice_scch4_magiccch4_us = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "scch4_30.csv"), data.table=FALSE)[,1]

# Load FUND SC-CH4 estimates that sample the U.S. climate sensivitiy distribution.
fund_scch4_fairch4_us   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "scch4_30.csv"), data.table=FALSE)[,1]
fund_scch4_fundch4_us   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "scch4_30.csv"), data.table=FALSE)[,1]
fund_scch4_hectorch4_us = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "scch4_30.csv"), data.table=FALSE)[,1]
fund_scch4_magiccch4_us = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "scch4_30.csv"), data.table=FALSE)[,1]

# Load DICE indices to ensure ECS sample and corresponding SC-CH4 estimates align (i.e. incase an individual parameter combination produced a model error).
dice_fairch4_indices   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fair", "good_indices.csv"), data.table=FALSE)[,1]
dice_fundch4_indices   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_fund", "good_indices.csv"), data.table=FALSE)[,1]
dice_hectorch4_indices = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_hector", "good_indices.csv"), data.table=FALSE)[,1]
dice_magiccch4_indices = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "dice", "s_magicc", "good_indices.csv"), data.table=FALSE)[,1]

# Load FUND indices to ensure ECS sample and corresponding SC-CH4 estimates align (i.e. incase an individual parameter combination produced a model error).
fund_fairch4_indices   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fair", "good_indices.csv"), data.table=FALSE)[,1]
fund_fundch4_indices   = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_fund", "good_indices.csv"), data.table=FALSE)[,1]
fund_hectorch4_indices = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_hector", "good_indices.csv"), data.table=FALSE)[,1]
fund_magiccch4_indices = fread(file.path("results", results_folder_name, "scch4_estimates", "us_climate_sensitivity", "fund", "s_magicc", "good_indices.csv"), data.table=FALSE)[,1]

# Load sampled U.S. climate sensivity values used for each SC-CH4 point estimate.
us_ecs_fairch4   = fread(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_fair", "ecs_sample.csv"), data.table=FALSE)[,1]
us_ecs_fundch4   = fread(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_fund", "ecs_sample.csv"), data.table=FALSE)[,1]
us_ecs_hectorch4 = fread(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_hector", "ecs_sample.csv"), data.table=FALSE)[,1]
us_ecs_magiccch4 = fread(file.path("results", results_folder_name, "climate_projections", "us_climate_sensitivity", "s_magicc", "ecs_sample.csv"), data.table=FALSE)[,1]

# Load posterior climate sensitivity values from each climate model calibration.
posterior_ecs_fairch4   = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_fair", "parameters_100k.csv"), data.table=FALSE)$ECS
posterior_ecs_fundch4   = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_fund", "parameters_100k.csv"), data.table=FALSE)$ECS
posterior_ecs_hectorch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_hector", "parameters_100k.csv"), data.table=FALSE)$ECS
posterior_ecs_magiccch4 = fread(file.path("results", results_folder_name, "calibrated_parameters", "s_magicc", "parameters_100k.csv"), data.table=FALSE)$ECS

#-------------------------------------------
# Climate Sensitivity Distributions
#-------------------------------------------

# Create a sample from U.S. climate sensitivity distribution.
us_ecs_sample = data.frame(us_ecs = 1.2 / (1 - rtnorm(500000, 00.6198, 0.1841, -0.2, 0.88)))

# Combine posterior climate sensitivity samples into dataframe.
posterior_ecs_samples = data.frame(fair=posterior_ecs_fairch4, fund=posterior_ecs_fundch4, hector=posterior_ecs_hectorch4, magicc=posterior_ecs_magiccch4)
colnames(posterior_ecs_samples) = c("fair", "fund", "hector", "magicc")

# Calculate means of each climate sensitivity sample.
ecs_means = data.frame(x=c(colMeans(posterior_ecs_samples), mean(us_ecs_sample$us_ecs)), y=rep(0, 5))

# Set some figure settings.
ecs_colors = c(fairch4_color, fundch4_color, hectorch4_color, magiccch4_color)
alpha = c(0.8, 0.8, 0.8, 0.8)
size = 0.4
x_range=c(0,10)
x_breaks = c(0,2,4,6,8,10)
x_labels=c("0", "2", "4", "6", "8", "10")
y_range = c(0,0.53)
shape=21
shape_size = 2.8

# Create plot of climate sensitivity distributions.
extended_fig_7a = ecs_pdf(posterior_ecs_samples, us_ecs_sample, alpha, ecs_colors, size, x_range, x_breaks, x_labels, y_range, expression(paste("Equilibrium Climate Sensitvity ("*degree*C*")")), c(0.0,0.2,0.5,0.1))

# Add points for mean climate sensitivity estimates.
extended_fig_7a = extended_fig_7a + geom_point(data=ecs_means, aes_string(x="x", y="y"), shape=8, size=1.5, stroke=0.25, colour=c(ecs_colors, "black"))

#-------------------------------------------------
# U.S. Climate Sensitivity & SC-CH4 Scatter Plots
#-------------------------------------------------

# Combine plot data for each model (for convenience).
dice_fairch4_data = data.frame(ecs = us_ecs_fairch4[dice_fairch4_indices], scch4 = dice_scch4_fairch4_us)
fund_fairch4_data = data.frame(ecs = us_ecs_fairch4[fund_fairch4_indices], scch4 = fund_scch4_fairch4_us)

dice_fundch4_data = data.frame(ecs = us_ecs_fundch4[dice_fundch4_indices], scch4 = dice_scch4_fundch4_us)
fund_fundch4_data = data.frame(ecs = us_ecs_fundch4[fund_fundch4_indices], scch4 = fund_scch4_fundch4_us)

dice_hectorch4_data = data.frame(ecs = us_ecs_hectorch4[dice_hectorch4_indices], scch4 = dice_scch4_hectorch4_us)
fund_hectorch4_data = data.frame(ecs = us_ecs_hectorch4[fund_hectorch4_indices], scch4 = fund_scch4_hectorch4_us)

dice_magiccch4_data = data.frame(ecs = us_ecs_magiccch4[dice_magiccch4_indices], scch4 = dice_scch4_magiccch4_us)
fund_magiccch4_data = data.frame(ecs = us_ecs_magiccch4[fund_magiccch4_indices], scch4 = fund_scch4_magiccch4_us)

# Create individual scatter plots
extended_fig_7b = us_ecs_scatter(dice_fairch4_data, fund_fairch4_data, 0.12, 1.5, 4000, c(0,5800), c(0,10.1), expression(paste("Equilibrium Climate Sensitvity ("*degree*C*")")), "Social Cost of Methane ($/t-CH4)", FALSE, TRUE, "S-FAIR", c(0.4, 5600), c(0.3,0.3,0.3,0.3))
extended_fig_7c = us_ecs_scatter(dice_fundch4_data, fund_fundch4_data, 0.12, 1.5, 4000, c(0,5800), c(0,10.1), expression(paste("Equilibrium Climate Sensitvity ("*degree*C*")")), "Social Cost of Methane ($/t-CH4)", FALSE, FALSE, "S-FUND", c(0.4, 5600), c(0.3,0.3,0.3,0.3))
extended_fig_7d = us_ecs_scatter(dice_hectorch4_data, fund_hectorch4_data, 0.12, 1.5, 4000, c(0,5800), c(0,10.1), expression(paste("Equilibrium Climate Sensitvity ("*degree*C*")", sep="")), "Social Cost of Methane ($/t-CH4)", TRUE, TRUE, "S-Hector", c(0.4, 5600), c(0.3,0.3,0.3,0.3))
extended_fig_7e = us_ecs_scatter(dice_magiccch4_data, fund_magiccch4_data, 0.12, 1.5, 4000, c(0,5800), c(0,10.1), expression(paste("Equilibrium Climate Sensitvity ("*degree*C*")")), "Social Cost of Methane ($/t-CH4)", TRUE, FALSE, "S-MAGICC", c(0.4, 5600), c(0.3,0.3,0.3,0.3))

# Create labeled panel for climate sensitivity distributions.
extended_fig_7_top = ggarrange(extended_fig_7a, labels=c("a"), label.args=list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=1.0))

# Combine all scatter plots into a single panel.
extended_fig_7_bottom  = ggarrange(extended_fig_7b, extended_fig_7c, extended_fig_7d, extended_fig_7e, nrow=2, ncol=2, labels= c("b", "c", "d", "e"), label.args = list(gp = grid::gpar(fontsize = 8, fontface="bold"), vjust=1))

# Merge all panels to create Extended Data Figure 7.
extended_fig_7 = grid.arrange(extended_fig_7_top, extended_fig_7_bottom, heights=c(1,2), nrow=2, ncol=1)

# Save a .jpg and .pdf version of Extended Data Figure 7.
ggsave(extended_fig_7, file=file.path("results", results_folder_name, "figures", "jpg_figures", "Extended_Data_Figure_7.jpg"), device="jpeg", type="cairo", width=130, height=180, unit="mm", dpi=300)
ggsave(extended_fig_7, file=file.path("results", results_folder_name, "figures", "pdf_figures", "Extended_Data_Figure_7.pdf"), device="pdf",  width=130, height=180, unit="mm", useDingbats = FALSE)

# Finished creating all figures.
print("All done.")
