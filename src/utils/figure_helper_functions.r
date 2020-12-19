# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------
# # This file contains various functions used to create figures.
# #-------------------------------------------------------------------------------------------------------
# #-------------------------------------------------------------------------------------------------------


#######################################################################################################################
# CLIMATE HINDCASTS & PROJECTIONS
#######################################################################################################################
# Description: This function will plot a climate projection spanning the 95% credible interval range, along with
#       `      a central model response and the calibration observations.
#
# Function Arguments:
#
#       conf_data       = Dataframe with annual climate projection information (mean response and lower/upper 95% ranges).
#       observations    = Dataframe holding the calibration observations.
#       obs_variable    = Column name in "observations" containing the appropriate data.
#       color           = Color to fill in the credible interval projection range.
#       y_range         = Span of y-axis.
#       x_range         = Span of x-axis.
#       y_title         = Title for the y-axis.
#       breaks          = Location for y-axis ticks/labels.
#       labels          = Label at each tick mark on the y-axis.
#       x_text          = TRUE/FALSE for whether or not to add x-axis label (needed for multi-panel figure).
#       y_text          = TRUE/FALSE for whether or not to add y-axis label (needed for multi-panel figure).
#       shape_size      = Size of points for observation data.
#       stroke          = Width of outline for each plotted observation point.
#       linesize_outer  = Width of line marking the 95% projection credible interval.
#       mean_width      = Width of line for the mean model response.
#       label           = String to add the name of the specific climate model into the corner of each plot.
#       label_xy        = x,y coordinates for where to locate the climate model label.
#       second_scenario = TRUE/FALSE for whether or not to add an overlapping climate projection for a second scenario.
#       conf_data_2     = Dataframe with annual climate projection information for the second scenario.
#       color_2         = Color for the 95% credible interval range for the second scenario.
#----------------------------------------------------------------------------------------------------------------------

climate_projection = function(conf_data, observations, obs_variable, color, y_range, x_range, x_title, y_title, breaks, labels, x_text, y_text, shape_size, stroke, linesize_outer, linetype_outer, mean_width, label, label_xy, second_scenario, conf_data_2, color_2){

    p = ggplot()

    # Add line to mark 0 value.
    p = p + geom_hline(yintercept=0, linetype="solid", colour="gray60", lwd=0.25)

    # If plotting two scenarios, plot the second hindcast/projection 95% interval.
    if(second_scenario == TRUE){
        p = p + geom_ribbon(data=conf_data_2, aes_string(x="Year", ymin="LowerConf_0.95", ymax="UpperConf_0.95"), fill=color_2)
    }

    # A hindcast/projections 95% interval, mean response, and calibration observations.
    p = p + geom_ribbon(data=conf_data, aes_string(x="Year", ymin="LowerConf_0.95", ymax="UpperConf_0.95"), fill=color)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="LowerConf_0.95"), colour="black", size=linesize_outer, linetype=linetype_outer)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="UpperConf_0.95"), colour="black", size=linesize_outer, linetype=linetype_outer)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="Mean"), colour="black", size=mean_width)
    p = p + geom_point(data=observations, aes_string(x="year", y=obs_variable), shape=21, size=shape_size, stroke=stroke, colour="black", fill="yellow", alpha=1)

    # Add model label.
    p = p + annotate("text", x=label_xy[1], y=label_xy[2], label = label, hjust = 0, vjust=1, size=8/(ggplot2:::.pt), fontface="italic", color="gray40")

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(expand = c(0, 0))
    p = p + scale_y_continuous(expand = c(0, 0), breaks=breaks, labels=labels)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle("")

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent", colour = "black", size=0.5),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_blank(),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.ticks = element_line(colour = "black", size = 0.25),
                  plot.margin = unit(c(0.4,0.1,0.1,0.2), "cm"))

    # Check for whether or not to add x-axis text/labels.
    if (x_text == TRUE){
        p = p + theme(axis.text.x = element_text(size=7, colour="black"),
                      axis.title.x = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.x = element_blank(),
                      axis.title.x = element_blank())
    }

    # Check for whether or not to add y-axis text/labels.
    if (y_text == TRUE){
        p = p + theme(axis.text.y = element_text(size=7, colour="black"),
                      axis.title.y = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.y = element_blank(),
                      axis.title.y = element_blank())
    }
}



#######################################################################################################################
# SC-CH4 DISTRIBUTIONS FOR ALL MODELS (SINGLE SCENARIO)
#######################################################################################################################
# Description: This function will plot eight SC-CH4 distributions (based on the four climate models coupled to DICE
#              and FUND).
#
# Function Arguments:
#
#       model_1    = A dataframe for climate model 1 with each column an IAM's SC-CH4 estimate.
#       model_2    = A dataframe for climate model 2 with each column an IAM's SC-CH4 estimate.
#       model_3    = A dataframe for climate model 3 with each column an IAM's SC-CH4 estimate.
#       model_4    = A dataframe for climate model 4 with each column an IAM's SC-CH4 estimate.
#       fund       = A dataframe with each column a climate model's SC-CH4 estimate for FUND (column names = "scch4_1", "scch4_2", ...).
#       alphas     = A vector of four values specifying the transparency of each SC-CH4 distribution.
#       colors     = A vector of four color names for each IAM's four SC-CH4 distributions.
#       size       = Width of distribution outline.
#       x_range    = Span of x-axis.
#       x_breaks   = Location for x-axis ticks/labels.
#       x_labels   = Label at each tick mark on the x-axis.
#       x_title    = Title for the x-axis.
#       y_range    = Span of y-axis.
#----------------------------------------------------------------------------------------------------------------------

scch4_pdf_baseline = function(model_1, model_2, model_3, model_4, alphas, colors, size, x_range, x_breaks, x_labels, x_title, y_range){

    p = ggplot()

    # Add four SC-CH4 distributions for FUND.
    p = p + geom_density(data=model_1, aes_string(x="fund"), color=colors[1], alpha=alphas[1], size=size)
    p = p + geom_density(data=model_2, aes_string(x="fund"), color=colors[2], alpha=alphas[2], size=size)
    p = p + geom_density(data=model_3, aes_string(x="fund"), color=colors[3], alpha=alphas[3], size=size)
    p = p + geom_density(data=model_4, aes_string(x="fund"), color=colors[4], alpha=alphas[4], size=size)

    # Add four SC-CH4 distributions for DICE.
    p = p + geom_density(data=model_1, aes_string(x="dice"), color=colors[1], alpha=alphas[1], size=size, linetype="22")
    p = p + geom_density(data=model_2, aes_string(x="dice"), color=colors[2], alpha=alphas[2], size=size, linetype="22")
    p = p + geom_density(data=model_3, aes_string(x="dice"), color=colors[3], alpha=alphas[3], size=size, linetype="22")
    p = p + geom_density(data=model_4, aes_string(x="dice"), color=colors[4], alpha=alphas[4], size=size, linetype="22")

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(clip="off", ylim=y_range, expand=FALSE)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels)
    p = p + xlab(x_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(colour = 'black', size = 0.35),
                  axis.text.x = element_text(size=7, colour="black"),
                  axis.title.x = element_text(size=7, colour="black"),
                  axis.ticks.x = element_line(colour = "black", size = 0.25),
                  axis.line.y = element_blank(),
                  axis.text.y = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  legend.position="none",
                  plot.title = element_blank())
}



#######################################################################################################################
# SC-CH4 DISTRIBUTIONS FOR VARIOUS SCENARIOS (SINGLE MODEL)
#######################################################################################################################
# Description: This function will plot four SC-CH4 distributions for a single climate model - IAM combination under four
#              different scenarios: (1) the main model specification, (2) using outdated CH4 radiative forcing equations
#              that neglect shortwave absorption, (3) removing the posterior parameter correlations, and (4) sampling the
#              equilibrium climate sensitivity distribution used for official U.S. SC-CH4 estimates.
#
# Function Arguments:
#
#       data1 - data4  = The SC-Ch4 estimates for each of the four scenarios (column should be named "scch4").
#       alpha          = A vector of four values specifying the transparency of each distribution's color.
#       colors         = A vector of four color names for the four distributions.
#       size           = Width of distribution outline.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#       x_title        = Title for the x-axis.
#       include_x_text = TRUE/FALSE for whether or not x-axis should have a title and tick mark labels.
#       y_range        = Span of y-axis.
#       line_type      = Type of line that outlines each distribution.
#       margins        = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

scch4_pdf_scenario = function(data1, data2, data3, data4, alpha, colors, size, x_range, x_breaks, x_labels, x_title, include_x_text, y_range, linetype, margins){

    p = ggplot()

    # Add four SC-CH4 distributions.
    p = p + geom_density(data=data1, aes_string(x="scch4"), color=colors[1], alpha=alpha[1], size=size, linetype=linetype)
    p = p + geom_density(data=data2, aes_string(x="scch4"), color=colors[2], alpha=alpha[2], size=size, linetype=linetype)
    p = p + geom_density(data=data3, aes_string(x="scch4"), color=colors[3], alpha=alpha[3], size=size, linetype=linetype)
    p = p + geom_density(data=data4, aes_string(x="scch4"), color=colors[4], alpha=alpha[4], size=size, linetype=linetype)

    # Scale axes and their positioning/labeling.
    p = p + coord_cartesian(clip="off", ylim=y_range, expand=FALSE)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(color="black", size=0.35),
                  axis.line.y = element_blank(),
                  axis.text.y = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  legend.position="none",
                  plot.title = element_blank(),
                  plot.margin = unit(margins, "cm"))

    # Check for whether or not to add x-axis text/labels.
    if (include_x_text == TRUE){
        p = p + xlab(x_title)
        p = p + theme(axis.text.x  = element_text(size=7, colour="black"),
                      axis.title.x = element_text(size=7, colour="black"),
                      axis.ticks.x = element_line(colour = "black", size = 0.25))
    } else {
        p = p + theme(axis.text.x  = element_blank(),
                      axis.title.x = element_blank(),
                      axis.ticks.x = element_blank())
    }
}



#######################################################################################################################
# IMPULSE RESPONSE SPAGHETTI PLOT FOR MULTIPLE SCENARIOS
#######################################################################################################################
# Description: This function will plot multiple impulse response model runs as semi-transparaent lines for four
#              different scenarios (in this case, discounted marginal climate damages over time).
#
# Function Arguments:
#
#       data1 - data4  = Dataframe of individual model runs for the four scenarios.
#       means          = Mean estimate over time for the four scenarios.
#       colors         = A vector of four color names for the four scenarios
#       alphas         = A vector of four values specifying the transparency of each line's color.
#       model_range    = Range of years the model was run for (used to isolate proper plotting data).
#       pulse_year     = Year that a one tonne methane emission pulse was added.
#       num_sims       = Number of model runs (lines) to plot for each scneario.
#       y_range        = Span of y-axis.
#       y_breaks       = Location for y-axis ticks/labels.
#       y_labels       = Label at each tick mark on the y-axis.
#       x_title        = Title for the x-axis.
#       y_title        = Title for the y-axis.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#----------------------------------------------------------------------------------------------------------------------

spaghetti_multi = function(data1, data2, data3, data4, means, colors, alphas, model_range, pulse_year, num_sims, y_range, y_breaks, y_labels, x_title, y_title, x_range, x_breaks, x_labels){

    # Get indices from model output for plotting range.
    start_index = which(model_range[1]:model_range[2] == (pulse_year-1))
    end_index   = which(model_range[1]:model_range[2] == x_range[2])

    # Isolate mean model responses for plotting range.
    mean_data = means[start_index:end_index, ]

    # Randomly sample results to plot "num_sims" model runs and melt the data together.
    rand_indices = sample(1:ncol(data1), num_sims)
    reshaped_data1 = reshape2::melt(data1[start_index:end_index,rand_indices])
    reshaped_data2 = reshape2::melt(data2[start_index:end_index,rand_indices])
    reshaped_data3 = reshape2::melt(data3[start_index:end_index,rand_indices])
    reshaped_data4 = reshape2::melt(data4[start_index:end_index,rand_indices])

    # Merge data into a data.frame (for plotting convenience).
    plot_data = data.frame(Period = rep((pulse_year-1):x_range[2], times=num_sims), reshaped_data1, reshaped_data2, reshaped_data3, reshaped_data4)
    colnames(plot_data) = c("Year", "Model_Run1", "Value1", "Model_Run2", "Value2", "Model_Run3", "Value3", "Model_Run4", "Value4")

    # Initialize plot.
    p = ggplot()

    # Add line to mark 0 value.
    p = p + geom_hline(yintercept=0, linetype="solid", colour="black", lwd=0.3)

    # Add spaghetti plots for four different scenarios (will plot "num_sims" lines for each scenario).
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value1", group="Model_Run1"), colour=colors[1], alpha=alphas[1])
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value2", group="Model_Run2"), colour=colors[2], alpha=alphas[2])
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value3", group="Model_Run3"), colour=colors[3], alpha=alphas[3])
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value4", group="Model_Run4"), colour=colors[4], alpha=alphas[4])

    # Add outlined mean responses for the four scenarios.
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean1"), colour="black", size=1.2)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean1"), colour=colors[1], size=0.6)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean2"), colour="black", size=1.2)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean2"), colour=colors[2], size=0.6)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean3"), colour="black", size=1.2)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean3"), colour=colors[3], size=0.6)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean4"), colour="black", size=1.2)
    p = p + geom_line(data=mean_data, aes_string(x="Year", y="Mean4"), colour=colors[4], size=0.6)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(ylim=y_range)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels, expand = c(0, 0))
    p = p + scale_y_continuous(breaks=y_breaks,labels=y_labels, expand = c(0, 0))
    p = p + xlab(x_title)
    p = p + ylab(y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(colour = 'black', size = 0.25),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.text = element_text(size=7, colour="black"),
                  axis.title = element_text(size=7),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.ticks = element_line(colour = "black", size = 0.25),
                  plot.margin = unit(c(0.1,0.15,0.2,0.2), "cm"))
    }



#######################################################################################################################
# ANNUAL DISTRIBUTIONS FOR INSET PANELS
#######################################################################################################################
# Description: This function will plot four different distributions (used for inset showing temperaure/marginal
#              damage pdfs for different years).
#
# Function Arguments:
#
#       data           = Values for each distribution and specific year being plotted (each column = different odf's data).
#       colors         = A vector of four color names for the four distributions.
#       alphas         = A vector of four transparency values for the four distributions.
#       linetypes      = Vector of line types to outline each distribution.
#       size           = Width of distribution outline.
#       alpha          = A vector of four values specifying the transparency of each distribution's color.
#       size           = Width of distribution outline.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#       x_title        = Title for the x-axis.
#       y_range        = Span of y-axis.
#       text_size      = Size of axis labels.
#       margins        = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

inset_4pdfs = function(data, colors, alphas, linetypes, size, x_range, x_breaks, x_labels, x_title, y_range, text_size, margins){

    p = ggplot()

    # Add distributions for the four years.
    p = p + geom_density(data=data, aes_string(x="Year_1"), fill=colors[1], alpha=alphas[1], size=size, linetype=linetypes[1])
    p = p + geom_density(data=data, aes_string(x="Year_2"), fill=colors[2], alpha=alphas[2], size=size, linetype=linetypes[2])
    p = p + geom_density(data=data, aes_string(x="Year_3"), fill=colors[3], alpha=alphas[3], size=size, linetype=linetypes[3])
    p = p + geom_density(data=data, aes_string(x="Year_4"), fill=colors[4], alpha=alphas[4], size=size, linetype=linetypes[4])

    # Scale axes and their positioning/labeling.
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels, expand = c(0, 0))
    p = p + scale_y_continuous(limits = y_range, expand = c(0, 0))

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  plot.background = element_rect(fill = "transparent", color="NA"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line.y = element_blank(),
                  axis.line.x = element_line(color="black", size=0.25),
                  axis.ticks.x = element_line(color="black", size=0.25),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  plot.margin = unit(margins, "cm"))

    # Check for whether or not to add x-axis text/labels.
    if(is.null(x_title)==TRUE){
        p = p + theme(axis.text.x = element_blank(),
                      axis.title.x = element_blank())
    } else {
        p = p + xlab(x_title)
        p = p + theme(axis.text.x = element_text(size=text_size, colour="gray40"),
                      axis.title.x = element_text(size=text_size, colour="gray40"))
    }
}



#######################################################################################################################
# ANNUAL DISTRIBUTIONS FOR INSET PANELS (TWO SCENARIOS).
#######################################################################################################################
# Description: This function will plot two annual distributions under two different scenarios (in this case, a
#              baseline case and a scenario without posterior correlations).
#
# Function Arguments:
#
#       data_base      = Values for two different years under the baseline scenario.
#       data_nocorr    = Values for two different years under the scenario without posterior parameter correlations.
#       colors         = A vector of two color names for each scenario.
#       alphas         = A vector of two transparency values for each scenario.
#       size           = Width of each distribution's outline.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#       x_title        = Title for the x-axis.
#       y_range        = Span of y-axis.
#       margins        = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

inset_nocorr_pdfs = function(data_base, data_nocorr, colors, alphas, size, x_range, x_breaks, x_labels, x_title, y_range, margins){

    p = ggplot()

    # Add distributions for each year-scenario pair.
    p = p + geom_density(data=data_base,   aes_string(x="Year_2"), fill=colors[1], alpha=alphas[1], size=size)
    p = p + geom_density(data=data_base,   aes_string(x="Year_1"), fill=colors[1], alpha=alphas[1], size=size)
    p = p + geom_density(data=data_nocorr, aes_string(x="Year_2"), fill=colors[2], alpha=alphas[2], size=size)
    p = p + geom_density(data=data_nocorr, aes_string(x="Year_1"), fill=colors[2], alpha=alphas[2], size=size)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(expand = FALSE, xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(breaks = x_breaks, labels=x_labels)
    p = p + xlab(x_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  plot.background = element_rect(fill = "transparent", color="NA"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line.y = element_blank(),
                  axis.line.x = element_line(color="black", size=0.25),
                  axis.ticks.x = element_line(color="black", size=0.25),
                  axis.text.x = element_text(size=7, colour="gray40"),
                  axis.title.x = element_text(size=7, colour="gray40"),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.text.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  plot.margin = unit(margins, "cm"))
}



#######################################################################################################################
# IMPULSE RESPONSE SPAGHETTI PLOT FOR SIGNLE SCENARIO
#######################################################################################################################
# Description: This function will plot multiple impulse response model runs as semi-transparaent lines for a single model.
#
# Function Arguments:
#
#       data           = Dataframe of individual model runs.
#       ci_data        = Dataframe with annual values (mean response and lower/upper 95% ranges).
#       color          = Color for each impulse response line.
#       alpha          = Value specifying the transparency of each line's color.
#       model_range    = Range of years the model was run for (used to isolate proper plotting data).
#       pulse_year     = Year that a one tonne methane emission pulse was added.
#       num_sims       = Number of model runs (lines) to plot for each scneario.
#       y_range        = Span of y-axis.
#       y_breaks       = Location for y-axis ticks/labels.
#       y_labels       = Label at each tick mark on the y-axis.
#       x_title        = Title for the x-axis.
#       y_title        = Title for the y-axis.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#----------------------------------------------------------------------------------------------------------------------

spaghetti_single = function(data, ci_data, color, alpha, model_range, pulse_year, num_sims, y_range, y_breaks, y_labels, x_title, y_title, x_range, x_breaks, x_labels){

    # Get indices from model output for plotting range.
    start_index = which(model_range[1]:model_range[2] == (pulse_year-1))
    end_index   = which(model_range[1]:model_range[2] == x_range[2])

    # Randomly sample results to plot "num_sims" model runs and melt the data together.
    rand_indices = sample(1:ncol(data), num_sims)
    reshaped_data = reshape2::melt(data[start_index:end_index,rand_indices])

    # Merge data into a data.frame (for plotting convenience).
    plot_data = data.frame(Period = rep((pulse_year-1):x_range[2], times=num_sims), reshaped_data)
    colnames(plot_data) = c("Year", "Model_Run", "Value")

    # Initialize plot.
    p = ggplot()

    # Add spaghetti plot (will plot "num_sims" lines) with mean response and 95% interval.
    p = p + geom_line(data=plot_data, aes_string(x="Year", y="Value", group="Model_Run"), colour=color, alpha=alpha)
    p = p + geom_line(data=ci_data, aes_string(x="Year", y="Lower_CI"), colour="black", linetype="dashed", size=0.2)
    p = p + geom_line(data=ci_data, aes_string(x="Year", y="Upper_CI"), colour="black", linetype="dashed", size=0.2)
    p = p + geom_line(data=ci_data, aes_string(x="Year", y="Mean"), colour="black", size=0.37)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(ylim=y_range)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels, expand = c(0, 0))
    p = p + scale_y_continuous(breaks=y_breaks,labels=y_labels, expand = c(0, 0))
    p = p + xlab(x_title)
    p = p + ylab(y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(colour = 'black', size = 0.25),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.text = element_text(size=7, colour="black"),
                  axis.title = element_text(size=7),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.ticks = element_line(colour = "black", size = 0.25),
                  plot.margin = unit(c(0.1,0.15,0.2,0.1), "cm"))
}



#######################################################################################################################
# CLIMATE PROJECTIONS FOR RCP2.6 SCENARIO
#######################################################################################################################
# Description: This function will plot a climate projection spanning the 95% credible interval range, along with
#       `      a central model response and the calibration observations for RCP 2.6.
#
# Function Arguments:
#
#       conf_data    = Dataframe with annual climate projection information (mean response and lower/upper 95% ranges).
#       observations = Dataframe holding the calibration observations.
#       obs_variable = Column name in "observations" containing the appropriate data.
#       color        = Color to fill in the credible interval projection range.
#       alpha        = Value specifying the transparency of the 95% credible interval range.
#       y_range      = Span of y-axis.
#       x_range      = Span of x-axis.
#       x_title      = Title for the x-axis.
#       y_title      = Title for the y-axis.
#       breaks       = Location for y-axis ticks/labels.
#       labels       = Label at each tick mark on the y-axis.
#       x_text       = TRUE/FALSE for whether or not to add x-axis label (needed for multi-panel figure).
#       y_text       = TRUE/FALSE for whether or not to add y-axis label (needed for multi-panel figure).
#       shape_size   = Size of points for observation data.
#       stroke       = Width of outline for each plotted observation point.
#       line_outer   = Width of line marking the 95% projection credible interval.
#       mean_width   = Width of line for the mean model response.
#       label        = String to add the name of the specific climate model into the corner of each plot.
#       label_xy     = x,y coordinates for where to locate the climate model label.
#       margin       = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

rcp26_projection = function(conf_data, observations, obs_variable, color, alpha, y_range, x_range, x_title, y_title, breaks, labels, x_text, y_text, shape_size, stroke, line_outer, mean_width, label, label_xy, margin){

    p = ggplot()

    # Add lines marking 1.5 and 2 degree C temperature targets.
    p = p + geom_hline(yintercept=1.5, colour="gray70", size=0.75)
    p = p + geom_hline(yintercept=2.0, colour="gray70", size=0.3)

    # Plot 95% projection interval and mean response.
    p = p + geom_ribbon(data=conf_data, aes_string(x="Year", ymin="LowerConf_0.95", ymax="UpperConf_0.95"), fill=color, alpha=alpha)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="LowerConf_0.95"), colour="black", size=line_outer)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="UpperConf_0.95"), colour="black", size=line_outer)
    p = p + geom_line(data=conf_data, aes_string(x="Year", y="Mean"), colour="black", size=mean_width)

    # Add observations and mark end of calibration period.
    p = p + geom_point(data=observations, aes_string(x="year", y=obs_variable), shape=21, size=shape_size, stroke=stroke, colour="black", fill="yellow", alpha=1)
    p = p + geom_vline(xintercept=2018, linetype="32", colour="red", lwd=0.25)

    # Add temperature target labels.
    p = p + annotate("text", x=label_xy[1], y=label_xy[2], label = label, hjust = 0, vjust=1, size=7/(ggplot2:::.pt), fontface="italic", color="gray40")
    p = p + annotate("text", x=2245, y=1.75, label = expression(paste("1.5"~degree*C*" target")), size=6/(ggplot2:::.pt), color="gray40")
    p = p + annotate("text", x=2245, y=2.25, label = expression(paste("2.0"~degree*C*" target")), size=6/(ggplot2:::.pt), color="gray40")

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(expand = c(0, 0))
    p = p + scale_y_continuous(expand = c(0, 0), breaks=breaks, labels=labels)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + ggtitle("")

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(colour = 'black', size = 0.25),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.ticks = element_line(colour = "black", size = 0.25),
                  plot.margin = unit(margin, "cm"))

    # Check for whether or not to add x-axis text/labels.
    if (x_text == TRUE){
        p = p + theme(axis.text.x = element_text(size=7, colour="black"),
                      axis.title.x = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.x = element_blank(),
                      axis.title.x = element_blank())
    }

    # Check for whether or not to add y-axis text/labels.
    if (y_text == TRUE){
        p = p + theme(axis.text.y = element_text(size=7, colour="black"),
                      axis.title.y = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.y = element_blank(),
                      axis.title.y = element_blank())
    }
}



#######################################################################################################################
# DISTRIBUTIONS FOR MULTIPLE EQUILIBRIUM CLIMATE SENSITIVITY SAMPLES
#######################################################################################################################
# Description: This function will plot distributions for the various equilibrium climate sensiivity samples.
#
# Function Arguments:
#
#       data     = Dataframe containing the posterior ECS samples (each column is a different model).
#       us_ecs   = Sample from the ECS distribution used for U.S. SC-CH4 estimates.
#       alphas   = Transparency value for each distribution line.
#       colors   = A vector of four color names for each model's sample.
#       size     = Width of each distribution's outline.
#       x_range  = Span of x-axis.
#       x_breaks = Location for x-axis ticks/labels.
#       x_labels = Label at each tick mark on the x-axis.
#       y_range  = Span of y-axis.
#       x_title  = Title for the x-axis.
#       margin   = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

ecs_pdf = function(data, us_ecs, alpha, colors, size, x_range, x_breaks, x_labels, y_range, x_title, margin){

    p = ggplot()

    # Add distributions for each ECS sample.
    p = p + stat_density(data=data, aes_string(x="fund"), geom="line", colour=colors[2], size=size)
    p = p + stat_density(data=data, aes_string(x="magicc"),geom="line",  colour=colors[4],  size=size)
    p = p + stat_density(data=data, aes_string(x="hector"),geom="line",  colour=colors[3], size=size)
    p = p + stat_density(data=data, aes_string(x="fair"), geom="line", colour=colors[1], size=size)
    p = p + stat_density(data=us_ecs, aes_string(x="us_ecs"), geom="line", colour="gray20", linetype="33", size=size)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(clip="off", ylim=y_range, expand=FALSE)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks, labels=x_labels)
    p = p + xlab(x_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(colour = 'black', size = 0.35),
                  axis.text.x = element_text(size=7, colour="black"),
                  axis.title.x = element_text(size=7, colour="black"),
                  axis.ticks.x = element_line(colour = "black", size = 0.25),
                  axis.line.y = element_blank(),
                  axis.text.y = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  legend.position="none",
                  plot.title = element_blank(),
                  plot.margin = unit(margin, "cm"))
}



#######################################################################################################################
# SCATTERPLOT OF SC-CH4 ESTIMATES AND U.S. CLIMATE SENSITIVITY SAMPLES
#######################################################################################################################
# Description: This function creates a scatter plot of U.S. climate sensitivity samples vs. the corresponding SC-CH4
#              estimates.
#
# Function Arguments:
#
#       dice_data    = Dataframe with DICE SC-CH4 estimates and the corresponding climate sensitivity values.
#       fund_data    = Dataframe with FUND SC-CH4 estimates and the corresponding climate sensitivity values.
#       alpha        = Value specifying the transparency of the points.
#       point_size   = Size of each plotted point.
#       n_points     = Total number of estimates to plot.
#       y_range      = Span of y-axis.
#       x_range      = Span of x-axis.
#       x_title      = Title for the x-axis.
#       y_title      = Title for the y-axis.
#       x_text       = TRUE/FALSE for whether or not to add x-axis label (needed for multi-panel figure).
#       y_text       = TRUE/FALSE for whether or not to add y-axis label (needed for multi-panel figure).
#       label        = String to add the name of the specific climate model into the corner of each plot.
#       label_xy     = x,y coordinates for where to locate the climate model label.
#       margin       = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#----------------------------------------------------------------------------------------------------------------------

us_ecs_scatter = function(dice_data, fund_data, alpha, point_size, n_points, y_range, x_range, x_title, y_title, x_text, y_text, label, label_xy, margin){

    # Get random samples of model output (will take "n_points" samples).
    rand_dice = sample(1:nrow(dice_data), n_points)
    rand_fund = sample(1:nrow(fund_data), n_points)

    # Initialize figure.
    p = ggplot()

    # Add scatter plot with fit line for DICE.
    p = p + geom_point(data=dice_data[rand_dice, ],  aes(x=ecs, y=scch4), shape=18, size=point_size, colour="black", alpha=alpha)
    p = p + geom_smooth(data=dice_data[rand_dice, ], aes_string(x="ecs", y="scch4"), method="loess", span=0.75, size=0.8, color="black", se=FALSE)
    p = p + geom_smooth(data=dice_data[rand_dice, ], aes_string(x="ecs", y="scch4"), method="loess", span=0.75, size=0.4, color="white", se=FALSE)

    # Add scatter plot with fit line for FUND.
    p = p + geom_point(data=fund_data[rand_fund, ],  aes(x=ecs, y=scch4), shape=16, size=point_size, colour="black", alpha=alpha)
    p = p + geom_smooth(data=fund_data[rand_fund, ], aes_string(x="ecs", y="scch4"), method="loess", span=0.75, size=0.8, color="black", se=FALSE)
    p = p + geom_smooth(data=fund_data[rand_fund, ], aes_string(x="ecs", y="scch4"), method="loess", span=0.75, size=0.4, color="white", se=FALSE)

    # Add label.
    p = p + annotate("text", x=label_xy[1], y=label_xy[2], label = label, hjust = 0, vjust=1, size=7/(ggplot2:::.pt), fontface="italic", color="gray40")

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(expand = c(0, 0), breaks = seq(0, 10, by=2.5), labels=c("0", "2.5", "5", "7.5", "10"))
    p = p + scale_y_continuous(expand = c(0, 0), breaks = seq(0, 5000, by=1000), labels=c("0", "1000", "2000", "3000", "4000", "5000"))
    p = p + xlab(x_title)
    p = p + ylab(y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.line = element_line(colour = 'black', size = 0.25),
                  legend.position="none",
                  plot.title = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.ticks = element_line(colour = "black", size = 0.25),
                  plot.margin = unit(margin, "cm"))

    # Check for whether or not to add x-axis text/labels.
    if (x_text == TRUE){
        p = p + theme(axis.text.x = element_text(size=7, colour="black"),
                      axis.title.x = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.x = element_blank(),
                      axis.title.x = element_blank())
    }

    # Check for whether or not to add y-axis text/labels.
    if (y_text == TRUE){
        p = p + theme(axis.text.y = element_text(size=7, colour="black"),
                      axis.title.y = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.y = element_blank(),
                      axis.title.y = element_blank())
    }
}



#######################################################################################################################
# SCATTERPLOT OF POSTERIOR CLIMATE MODEL PARAMETERS AND SC-CH4 ESTIMATES
#######################################################################################################################
# Description: This function creates a scatter plot of various model parameters and the corresponding SC-CH4
#              estimates (will adjust size and color of each point based on value of different parameters).
#
# Function Arguments:
#
#       plot_data      = Dataframe with different model parameter samples and corresponding SC-CH4 estimates.
#       var_names      = Vector of column names in "plot_data" that will be depicted in the figure.
#       n_points       = Total number of estimates to plot.
#       shape          = Shape to use for each point.
#       span           = Controls width of moving window in stat_smooth.
#       colors         = Vector of color names to create continuous transition that colors points based on parameter values.
#       color_limits   = Range of values for upper/lower bound for color values.
#       color_breaks   = Breaks for color labels.
#       color_labels   = Vector for values at color breaks.
#       size_limits    = Range of values for upper/lower bound for size values.
#       size_breaks    = Breaks for size labels.
#       size_labels    = Vector for values at size breaks.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#       y_range        = Span of y-axis.
#       y_breaks       = Location for y-axis ticks/labels.
#       y_labels       = Label at each tick mark on the y-axis.
#       x_title        = Title for the x-axis.
#       y_title        = Title for the y-axis.
#       legend_title   = Title for legend.
#       margin         = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#       second_scatter = TRUE/FALSE for whether or not to plot a second scatter plot for a different model/scenario.
#       var_names2     = Vector of column names in "plot_data" for second scenario/model that will be depicted in the figure.
#       shape2         = Shape to use for each point for a second model/scenario.
#       x_text         = TRUE/FALSE for whether or not to add x-axis label (needed for multi-panel figure).
#       y_text         = TRUE/FALSE for whether or not to add y-axis label (needed for multi-panel figure).
#----------------------------------------------------------------------------------------------------------------------

scatter_4way = function(plot_data, var_names, n_points, shape, span, colors, color_limits, color_breaks, color_labels, size_limits, size_breaks, size_labels, x_range, x_breaks, x_labels, y_range, y_breaks, y_labels, x_title, y_title, legend_titles, margin, second_scatter, var_names2, shape2, x_text, y_text){

    # Create random indices to sample a subset of data to plot (will take "n_points" samples).
    thin_indices = sample(1:nrow(plot_data), n_points)

    # Initialize figure.
    p = ggplot()

    # Add scatter plot with fit line.
    p = p + geom_point(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2], size=var_names[3], fill=var_names[4]), shape=shape, stroke=0.25)
    p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2]), method="loess", span=span, size=0.8, color="black", se=FALSE, fullrange=TRUE)
    p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2]), method="loess", span=span, size=0.3, color="white", se=FALSE, fullrange=TRUE)

    # If also plotting a second model/scenario, add that scatter plot with fit line.
    if(second_scatter==TRUE){
        p = p + geom_point(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2], size=var_names2[3], fill=var_names2[4]), shape=shape2, stroke=0.25)
        p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2]), method="loess", span=span, size=0.8, color="black", se=FALSE, fullrange=TRUE)
        p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2]), method="loess", span=span, size=0.3, color="white", se=FALSE, fullrange=TRUE)
    }

    # Set conditions on point colors and size.
    p = p + scale_fill_gradientn(colours=colors, limits=color_limits, breaks=color_breaks, labels=color_labels, oob=scales::squish,  guide = guide_colorbar(frame.colour = "black", ticks = FALSE,  frame.linewidth = 0.5, barheight=2.5, barwidth=0.4))
    p = p + scale_size_continuous(range=size_limits, breaks=size_breaks, labels=size_labels)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(expand = FALSE, xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(breaks = x_breaks, labels=x_labels)
    p = p + scale_y_continuous(breaks = y_breaks, labels=y_labels)
    p = p + xlab(x_title)
    p = p + ylab(y_title)
    p = p + labs(fill=legend_titles[1])
    p = p + labs(size=legend_titles[2])

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(colour = 'black', size = 0.35),
                  axis.ticks.x = element_line(colour = "black", size = 0.25),
                  axis.line.y = element_line(colour = 'black', size = 0.35),
                  axis.ticks.y = element_line(colour = "black", size = 0.25),
                  plot.title = element_blank(),
                  legend.position = c(0.45, 0.85),
                  legend.text = element_text(size=6),
                  legend.title = element_text(size=6),
                  legend.background=element_blank(),
                  legend.key=element_blank(),
                  legend.key.size = unit(4.0, "mm"),
                  legend.box = "horizontal",
                  plot.margin = unit(margin, "mm"))

    # Check for whether or not to add x-axis text/labels.
    if (x_text == TRUE){
        p = p + theme(axis.text.x = element_text(size=7, colour="black"),
                      axis.title.x = element_text(size=7, colour="black"))
    } else {
        p = p + theme(axis.text.x = element_blank(),
                      axis.title.x = element_blank())
    }

    # Check for whether or not to add y-axis text/labels.
    if (y_text == TRUE){
        p = p + theme(axis.text.y = element_text(size=7, colour="black"),
                      axis.title.y = element_text(size=7, colour="black"))
      } else {
                 p = p + theme(axis.text.y = element_blank(),
                              axis.title.y = element_blank())
      }
}

# Small function to adjust axis label spacing (note*: found this code snippet online).
draw_key_polygon3 <- function(data, params, size) {
  lwd <- min(data$size, min(size) / 4)

  grid::rectGrob(
    width = grid::unit(0.6, "npc"),
    height = grid::unit(0.6, "npc"),
    gp = grid::gpar(
    col = data$colour,
    fill = alpha(data$fill, data$alpha),
    lty = data$linetype,
    lwd = lwd * .pt,
    linejoin = "mitre"
    ))
}

# Register function.
GeomBar$draw_key = draw_key_polygon3



#######################################################################################################################
# EQUITY-WEIGHTED SC-CH4 ESTIMATES FOR DIFFERENT REGIONS (95% INTERVALS)
#######################################################################################################################
# Description: This function will plot the 95% credible intervals for different world regions across a range of
#              elasticity of marginal utility of consumption values.
#
# Function Arguments:
#
#       scch4_equity  = Dataframe of central and 95% interval values for regional equity-weighted SC-CH4 estimates.
#       regions       = Vector of region names to plot.
#       color         = Vector of colors to fill each credible interval projection range.
#       alpha         = Vector of values specifying the transparency of the 95% credible interval range.
#       axis_settings = List of various axis settings.
#       line_size     = Thickness of lines.
#       margin        = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#       vline         = TRUE/FALSE for whether or not to plot a vertical line highlighting a particular elasticity value.
#       vline_value   = X-axis value to place vertical line.
#       zoom_in       = TRUE/FALSE for whether or not the plot depicts a zoomed in view of lower equity-weighted SC-CH4 estimates.
#----------------------------------------------------------------------------------------------------------------------

equity_ci = function(scch4_equity, regions, colors, alphas, axis_settings, line_size, margin, vline, vline_value, zoom_in){

    p = ggplot()

    # Check for whether or not to add a vertical line.
    if(vline==TRUE){
        p = p + geom_vline(xintercept=vline_value, color="red", size=0.3)
    }

    # Add 95% interval and mean response for Region 1.
    p = p + geom_ribbon(data=scch4_equity[[regions[1]]], aes_string(x="eta", ymin="lower_ci", ymax="upper_ci"), fill=colors[1], alpha=alphas[1])
    p = p + geom_line(data=scch4_equity[[regions[1]]],   aes_string(x="eta", y="mean"), size=line_size)
    p = p + geom_line(data=scch4_equity[[regions[1]]],   aes_string(x="eta", y="mean"), size=line_size*0.7, colour=colors[1])

    # Add 95% interval and mean response for Region 2.
    p = p + geom_ribbon(data=scch4_equity[[regions[2]]], aes_string(x="eta", ymin="lower_ci", ymax="upper_ci"), fill=colors[2], alpha=alphas[2])
    p = p + geom_line(data=scch4_equity[[regions[2]]],   aes_string(x="eta", y="mean"),size=line_size)
    p = p + geom_line(data=scch4_equity[[regions[2]]],   aes_string(x="eta", y="mean"), size=line_size*0.7, colour=colors[2])

    # Add 95% interval and mean response for Region 3.
    p = p + geom_ribbon(data=scch4_equity[[regions[3]]], aes_string(x="eta", ymin="lower_ci", ymax="upper_ci"), fill=colors[3], alpha=alphas[3])
    p = p + geom_line(data=scch4_equity[[regions[3]]],   aes_string(x="eta", y="mean"),size=line_size)
    p = p + geom_line(data=scch4_equity[[regions[3]]],   aes_string(x="eta", y="mean"), size=line_size*0.7, colour=colors[3])

    # Add 95% interval and mean response for Region 4.
    p = p + geom_ribbon(data=scch4_equity[[regions[4]]], aes_string(x="eta", ymin="lower_ci", ymax="upper_ci"), fill=colors[4], alpha=alphas[4])
    p = p + geom_line(data=scch4_equity[[regions[4]]],   aes_string(x="eta", y="mean"),size=line_size)
    p = p + geom_line(data=scch4_equity[[regions[4]]],   aes_string(x="eta", y="mean"), size=line_size*0.7, colour=colors[4])

    # Add 95% interval and mean response for Region 5.
    p = p + geom_ribbon(data=scch4_equity[[regions[5]]], aes_string(x="eta", ymin="lower_ci", ymax="upper_ci"), fill=colors[5], alpha=alphas[5])
    p = p + geom_line(data=scch4_equity[[regions[5]]],   aes_string(x="eta", y="mean"),size=line_size)
    p = p + geom_line(data=scch4_equity[[regions[5]]],   aes_string(x="eta", y="mean"), size=line_size*0.7, colour=colors[5])

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(expand = FALSE, xlim=axis_settings$x_lim, ylim=axis_settings$y_lim)
    p = p + scale_x_continuous(breaks = axis_settings$x_breaks, labels=axis_settings$x_labels)
    p = p + scale_y_continuous(breaks = axis_settings$y_breaks, labels=axis_settings$y_labels)
    p = p + xlab(axis_settings$x_title)
    p = p + ylab(axis_settings$y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.text = element_text(size=7, colour="black"),
                  axis.title = element_text(size=7, colour="black"),
                  plot.title = element_blank(),
                  legend.position="none",
                  plot.margin = unit(margin, "mm"))

    # Check for whether or not this plot corresponds to a zoomed in portion of the equity-weighted estimates.
    if(zoom_in==TRUE){
        p = p + theme(panel.border = element_rect(colour = "gray50", fill=NA, size=0.45, linetype="22"),
                      axis.ticks   = element_line(colour = "gray50", size = 0.25))
    } else {
        p = p + theme(axis.line = element_line(colour = 'black', size = 0.35),
                      axis.ticks   = element_line(colour = "black", size = 0.25))
    }
}



#######################################################################################################################
# EQUITY-WEIGHTED SC-CH4 DISTRIBUTIONS FOR A SINGLE ELASTICITY OF MARGINAL UTILITY OF CONSUMPTION VALUE
#######################################################################################################################
# Description: This function will plot equity-weighted SC-CH4 distributions for different world regions for a single
#              elasticity of marginal utility of consumption.
#
# Function Arguments:
#
#       equity_data   = Dataframe of equity-weighted SC-CH4 point estimates for different world regions.
#       colors        = Vector of colors to fill each distribution.
#       alphas        = Vector of values specifying the transparency of each distribution.
#       axis_settings = List of various axis settings.
#       line_size     = Thickness of each distribution's outline.
#       margin        = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#       border_color  = Color to outline figure in.
#       vline         = TRUE/FALSE for whether or not to plot a vertical line highlighting a particular SC-CH4 value.
#       vline_value   = X-axis value to place vertical line.
#----------------------------------------------------------------------------------------------------------------------

equity_pdfs = function(equity_data, colors, alphas, axis_settings, line_size, margin, border_color, vline, vline_value){

    p = ggplot()

    # Add option for a vertical line.
    if(vline==TRUE){
        p = p + geom_vline(xintercept=vline_value, color="gray50", size=0.3)
    }

    # Add four SC-CH4 distributions.
    p = p + geom_density(aes(x=equity_data$region1), fill=colors[1], alpha=alphas[1], size=line_size)
    p = p + geom_density(aes(x=equity_data$region2), fill=colors[2], alpha=alphas[2], size=line_size)
    p = p + geom_density(aes(x=equity_data$region3), fill=colors[3], alpha=alphas[3], size=line_size)
    p = p + geom_density(aes(x=equity_data$region4), fill=colors[4], alpha=alphas[4], size=line_size)
    p = p + geom_density(aes(x=equity_data$region5), fill=colors[5], alpha=alphas[5], size=line_size)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(expand = FALSE, ylim=axis_settings$y_lim, axis_settings$x_lim)
    p = p + scale_x_continuous(limits = axis_settings$x_lim, breaks = axis_settings$x_breaks, labels=axis_settings$x_labels)
    p = p + scale_y_continuous(breaks = axis_settings$y_breaks, labels=axis_settings$y_labels)
    p = p + xlab(axis_settings$x_title)
    p = p + ylab(axis_settings$y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.text = element_text(size=7, colour="black"),
                  axis.title = element_text(size=7, colour="black"),
                  panel.border = element_rect(colour = border_color, fill=NA, size=0.4),
                  axis.ticks = element_line(colour = border_color, size = 0.25),
                  axis.line = element_blank(),
                  plot.title = element_blank(),
                  legend.position="none",
                  plot.margin = unit(margin, "mm"))
}



#######################################################################################################################
# SC_CH4 DISTRIBUTIONS FOR WIDER PRIOR PARAMETERS
#######################################################################################################################
# Description: This function will plot distributions for the basline scenario and the scenario using wider prior
#              parameter distributions during model calibration (pools results across models for DICE and FUND).
#
# Function Arguments:
#
#       dice       = SC-CH4 estimates for DICE under baseline scenario.
#       fund       = SC-CH4 estimates for FUND under baseline scenario.
#       wide_dice  = SC-CH4 estimates for DICE under wider priors.
#       wide_fund  = SC-CH4 estimates for FUND under wider priors.
#       base_color = Color for baseline distributions.
#       wide_color = Color for wider prior distributions.
#       size       = Width of distribution lines.
#       alpha      = Value specifying the transparency of each distribution.
#       x_range    = Span of x-axis.
#       x_breaks   = Location for x-axis ticks/labels.
#       x_labels   = Label at each tick mark on the x-axis.
#       x_title    = Title for the x-axis.
#       y_range    = Span of y-axis.
#----------------------------------------------------------------------------------------------------------------------

scch4_wider_pdfs = function(dice, fund, wide_dice, wide_fund, base_color, wide_color, size, alpha, x_range, x_breaks, x_labels, x_title, y_range){

    p = ggplot()

    # Add SC-CH4 distributions for DICE.
    p = p + stat_density(data=dice, geom="line", aes_string(x="scch4"), colour=base_color, alpha=alpha, size=size, linetype="22")
    p = p + stat_density(data=wide_dice, geom="line", aes_string(x="scch4"), colour=wide_color, alpha=alpha, size=size, linetype="22")

    # Add SC-CH4 distributions for FUND.
    p = p + stat_density(data=fund, geom="line", aes_string(x="scch4"), colour=base_color, alpha=alpha, size=size)
    p = p + stat_density(data=wide_fund, geom="line", aes_string(x="scch4"), colour=wide_color, alpha=alpha, size=size)

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(ylim=y_range, expand=FALSE)
    p = p + scale_x_continuous(limits = x_range, breaks=x_breaks,labels=x_labels)
    p = p + xlab(x_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(colour = 'black', size = 0.35),
                  axis.text.x = element_text(size=7, colour="black"),
                  axis.title.x = element_text(size=7, colour="black"),
                  axis.ticks.x = element_line(colour = "black", size = 0.25),
                  axis.line.y = element_blank(),
                  axis.text.y = element_blank(),
                  axis.title.y = element_blank(),
                  axis.ticks.y = element_blank(),
                  legend.position="none",
                  plot.title = element_blank())
}



#######################################################################################################################
# SCATTERPLOT OF SC-CH4 ESTIMATES AND METHANE CYCLE PARAMETERS
#######################################################################################################################
# Description: This function creates a scatter plot of methane cycle parameters and the corresponding SC-CH4
#              estimates.
#
# Function Arguments:
#
#       plot_data      = Dataframe of posterior methane cycle parameters and the corresponding SC-CH4 estimates.
#       var_names      = Column names in "plot_data".
#       n_points       = Total number of estimates to plot.
#       shape          = Shape to use for each point.
#       span           = Controls width of moving window in stat_smooth.
#       color          = Point color.
#       alpha          = Value specifying the transparency of the points.
#       size_limits    = Range of values for upper/lower bound for size values.
#       size_breaks    = Breaks for size labels.
#       size_labels    = Vector for values at size breaks.
#       x_range        = Span of x-axis.
#       x_breaks       = Location for x-axis ticks/labels.
#       x_labels       = Label at each tick mark on the x-axis.
#       y_range        = Span of y-axis.
#       y_breaks       = Location for y-axis ticks/labels.
#       y_labels       = Label at each tick mark on the y-axis.
#       x_title        = Title for the x-axis.
#       y_title        = Title for the y-axis.
#       legend_title   = Title for legend.
#       margin         = Margin around entire plot (ordered by the sizes of the top, right, bottom, and left margins).
#       fixed_size     = TRUE/FALSE for whether or not the size of each point should vary with a parameter's value.
#       second_scatter = TRUE/FALSE for whether or not to plot estimates from a second model/scenario.
#       var_names2     = Column names in "plot_data" for second model/scenario.
#       shape2         = Shape to use for each point for second model/scenario.
#       color2         = Point color for second model/scenario.
#----------------------------------------------------------------------------------------------------------------------

ch4cycle_scatter = function(plot_data, var_names, n_points, shape, span, color, alpha, size_limits, size_breaks, size_labels, x_range, x_breaks, x_labels, y_range, y_breaks, y_labels, x_title, y_title, legend_title, margin, fixed_size, size, second_scatter, var_names2, shape2, color2){

    # Create random indices to sample a subset of data (will take "n_points" samples).
    thin_indices = sample(1:nrow(plot_data), n_points)

    # Initialize figure.
    p = ggplot()

    # Add scatter plot, with check for whether or not point size should depend on a climate parameter value.
    if(fixed_size==TRUE){
        p = p + geom_point(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2]), size=size, fill=color, shape=shape, stroke=0.25, alpha=alpha)
    } else {
        p = p + geom_point(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2], size=var_names[3]), fill=color, shape=shape, stroke=0.25)
        p = p + labs(size=legend_title)
        p = p + scale_size_continuous(range=size_limits, breaks=size_breaks, labels=size_labels)
    }

    # Add fit line.
    p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2]), method="loess", span=span, size=0.8, color="black", se=FALSE, fullrange=TRUE)
    p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names[1], y=var_names[2]), method="loess", span=span, size=0.3, color="white", se=FALSE, fullrange=TRUE)

    # Check for whether or not to add a second scatter plot for another model/scenario.
    if(second_scatter==TRUE){
        p = p + geom_point(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2], size=var_names2[3]), fill=color2, shape=shape2, stroke=0.25)
        p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2]), method="loess", span=span, size=0.8, color="black", se=FALSE, fullrange=TRUE)
        p = p + stat_smooth(data=plot_data[thin_indices, ], aes_string(x=var_names2[1], y=var_names2[2]), method="loess", span=span, size=0.3, color="white", se=FALSE, fullrange=TRUE)
    }

    # Scale axes and their positioning/labeling + add titles.
    p = p + coord_cartesian(expand = FALSE, xlim=x_range, ylim=y_range)
    p = p + scale_x_continuous(breaks = x_breaks, labels=x_labels)
    p = p + scale_y_continuous(breaks = y_breaks, labels=y_labels)
    p = p + xlab(x_title)
    p = p + ylab(y_title)

    # Set plot theme/style.
    p = p + theme(panel.background = element_rect(fill = "transparent"),
                  panel.grid.minor = element_blank(),
                  panel.grid.major = element_blank(),
                  axis.ticks.length=unit(.1, "cm"),
                  axis.line.x = element_line(colour = 'black', size = 0.35),
                  axis.text.x = element_text(size=7, colour="black"),
                  axis.title.x = element_text(size=7, colour="black"),
                  axis.ticks.x = element_line(colour = "black", size = 0.25),
                  axis.line.y = element_line(colour = 'black', size = 0.35),
                  axis.text.y = element_text(size=7, colour="black"),
                  axis.title.y = element_text(size=7, colour="black"),
                  axis.ticks.y = element_line(colour = "black", size = 0.25),
                  plot.title = element_blank(),
                  legend.position = c(0.45, 0.85),
                  legend.text = element_text(size=6),
                  legend.title = element_text(size=6),
                  legend.background=element_blank(),
                  legend.key=element_blank(),
                  legend.key.size = unit(4.0, "mm"),
                  legend.box = "horizontal",
                  plot.margin = unit(margin, "mm"))
}
