library(earlywarnings)
library(ggplot2)
library(dplyr)
library(tidyr)


#----------------------------------------------

#               MANAUS

#----------------------------------------------

#investigating EWS before the identified break points through the library bfast (script: bfast_analysis.R)

##############################################
#       BREAK POINT FROM 1 YEAR FREQUENCY
#                  1) 1988-06
##############################################

#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 1 year
#   1st breakpoint: 1988-06
#   2nd breakpoint: 1999-02
#   variable: NPP
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_1y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_1y_monthly.csv")
# 
# #selecting before 1st breakpoint:
df_1y_1bp <- df_1y[df_1y$Date < "1988-06",]

# #selecting before 1st breakpoint:
df_1y_2bp <- df_1y[df_1y$Date < "1999-02",]

df_1y_3bp <- df_1y[df_1y$Date > "1999-02",]


# 
#select the variable of interest
df_1y_1bp_npp <- df_1y_1bp$Monthly_NPP_Mean

#select the variable of interest
df_1y_2bp_npp <- df_1y_2bp$Monthly_NPP_Mean

df_1y_3bp_npp <- df_1y_3bp$Monthly_NPP_Mean

# 

# Aplicar generic early warning signals
df_1y_bp_npp_gws_1bp <- generic_ews(df_1y_1bp_npp, winsize = 20, detrending = 'loess', 
                                logtransform = FALSE, interpolate = FALSE, 
                                AR_n = FALSE, powerspectrum = FALSE)

# Aplicar generic early warning signals
df_1y_bp_npp_gws_2bp <- generic_ews(df_1y_2bp_npp, winsize = 20, detrending = 'loess', 
                                    logtransform = FALSE, interpolate = FALSE, 
                                    AR_n = FALSE, powerspectrum = FALSE)

##Take the Nas values?
df_1y_bp_npp_gws_3bp <- generic_ews(df_1y_3bp_npp, winsize = 50, detrending = 'loess', 
                                    logtransform = FALSE, interpolate = FALSE, 
                                    AR_n = FALSE, powerspectrum = FALSE)



df_1y_bp_npp_gws$frequency = "1"


write.csv(df_1y_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_1y_bp1y_ews.csv", row.names = FALSE)
#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 3 years
#   1st breakpoint: 1987-08
#   variable: NPP
#----------------------------------------------

# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_3y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_3y_monthly.csv")
# 
# #selecting before 1st breakpoint:
df_3y_bp <- df_3y[df_3y$Date < "1987-08",]
# 
df_3y_bp_npp <- df_3y_bp$Monthly_NPP_Mean
# 
df_3y_bp_npp_gws <- generic_ews(df_3y_bp_npp, winsize = 50, detrending = 'gaussian',
                                 bandwidth = 10, logtransform = FALSE,interpolate = FALSE,
                                 AR_n = FALSE, powerspectrum = TRUE)

df_3y_bp_npp_gws$frequency = "3"

# 
write.csv(df_3y_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_3y_bp1y_ews.csv", row.names = FALSE)


#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 5 years
#   1st breakpoint: 1987-08
#   variable: NPP
#----------------------------------------------
df_5y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_5y_monthly.csv")
# 
# #selecting before 1st breakpoint:
df_5y_bp <- df_5y[df_5y$Date < "1987-08",]
# 
df_5y_bp_npp <- df_5y_bp$Monthly_NPP_Mean
# 
df_5y_bp_npp_gws <- generic_ews(df_5y_bp_npp, winsize = 50, detrending = 'gaussian',
                                bandwidth = 10, logtransform = FALSE,interpolate = FALSE,
                                AR_n = FALSE, powerspectrum = TRUE)

df_5y_bp_npp_gws$frequency = "5"

# 
write.csv(df_5y_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_5y_bp1y_ews.csv", row.names = FALSE)

#----------------------------------------------
#   precipitation reduction: 30%
#   frequency: 7 years
#   1st breakpoint: 1987-08
#   variable: NPP
#----------------------------------------------
# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_7y <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_30prec_7y_monthly.csv")
# 
# #selecting before 1st breakpoint:
df_7y_bp <- df_7y[df_7y$Date < "1987-08",]
# 
df_7y_bp_npp <- df_7y_bp$Monthly_NPP_Mean
# 
df_7y_bp_npp_gws <- generic_ews(df_7y_bp_npp, winsize = 50, detrending = 'gaussian',
                                bandwidth = 10, logtransform = FALSE,interpolate = FALSE,
                                AR_n = FALSE, powerspectrum = TRUE)

df_7y_bp_npp_gws$frequency = "7"

# 
write.csv(df_7y_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_7y_bp1y_ews.csv", row.names = FALSE)
# 
# # ----------------------------------------------
# #   precipitation reduction: regular climate
# #   frequency: regular climate
# #   1st breakpoint: 1987-08
# #   variable: NPP
# # ----------------------------------------------
# # !!!!! note this is the monthly integrated data frame!!!!!!!
df_regularclimate <- read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/src/MAN_regularclimate_monthly.csv")
# 
# #selecting before 1st breakpoint:
df_regularclimate_bp <- df_regularclimate[df_regularclimate$Date < "1987-08",]
# 
df_regularclimate_bp_npp <- df_regularclimate_bp$Monthly_NPP_Mean
# 
df_regularclimate_bp_npp_gws <- generic_ews(df_regularclimate_bp_npp, winsize = 50, detrending = 'gaussian',
                                bandwidth = 10, logtransform = FALSE,interpolate = FALSE,
                                AR_n = FALSE, powerspectrum = TRUE)

df_regularclimate_bp_npp_gws$frequency = "regularclimate"

# 
write.csv(df_regularclimate_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_regularclimate_bp1y_ews.csv", row.names = FALSE)
# 
#
#concat the csvs with all frequencies
df_allfreq_bp_npp_gws = rbind(df_regularclimate_bp_npp_gws,df_7y_bp_npp_gws,df_5y_bp_npp_gws,
                              df_3y_bp_npp_gws, df_1y_bp_npp_gws)


write.csv(df_allfreq_bp_npp_gws, file =
            "/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_allfreq_bp1y_ews.csv", row.names = FALSE)

# ploting the ar(1) according to the frequency of application
ar_allfreq = read.csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/MAN_30prec_allfreq_bp1y_ews.csv")


# Crie o gráfico usando ggplot2
ggplot(ar_allfreq, aes(x = timeindex, y = ar1)) 






# ##############################################
# #       BREAK POINT FROM 3 YEAR FREQUENCY
# #                  1991-11
# ##############################################
# 
# # ----------------------------------------------
# #   precipitation reduction: regular climate
# #   frequency: regular climate
# #   1st breakpoint: 1991-11
# #   variable: NPP
# # ----------------------------------------------
# 
# 
# # # !!!!! note this is the monthly integrated data frame!!!!!!!
# # df_regularclimate <- read.csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/MAN_regularclimate/gridcell186-239/MAN_regularclimate_monthly.csv")
# 
# # # #selecting before 1st breakpoint:
# # df_regularclimate_bp <- df_regularclimate[df_regularclimate$Date < "1991-11",]
# 
# # df_regularclimate_bp_npp <- df_regularclimate_bp$Monthly_NPP_Mean

# df_regularclimate_bp_npp_gws <- generic_ews(df_regularclimate_bp_npp, winsize = 10, detrending = 'gaussian',
#                                 bandwidth = 10, logtransform = FALSE,interpolate = FALSE,
#                                 AR_n = TRUE, powerspectrum = FALSE)

# df_regularclimate_bp_npp_gws$ar1

# plot(df_regularclimate_bp_npp_gws)
# first_plot = recordPlot()
# # # Create or display the plot
# png("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/ews_30perc_before_3ybp_regclim.png")
# replayPlot(first_plot)
# dev.off()
