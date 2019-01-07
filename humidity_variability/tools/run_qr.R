#!/usr/bin/env Rscript

# Run quantile regression (standard) on an input dataset for specified quantiles.
#
# Parameters
# ----------
# data_fname : str
#     Full path and filename for dataset to analyze.
# lq : int
#     Lowest quantile to run analysis on [0, 100]
# uq : int
#     Highest quantile to run analsis on [0, 100]
# qstep : int
#     Step to take between quantiles
#
#
# Returns
# -------
# Nothing
#
# Saves
# -----
# New csv file with quantile output


library("optparse")
library("quantreg")

option_list = list(
    make_option(c("-f", "--file"), type="character", default=NULL,
                help="csv file containing predictor and predictand", metavar="character"),
    make_option(c("-x", "--predictor_name"), type="character", default=NULL,
                help="column name for predictor", metavar="character"),
    make_option(c("-y", "--predictand_name"), type="character", default=NULL,
                help="column name for predictand", metavar="character"),
    make_option(c("-l", "--lower_quant"), type="integer", default=5,
                help="lowest quantile [default %default]", metavar="number"),
    make_option(c("-u", "--upper_quant"), type="integer", default=95,
                help="highest quantile [default %default]", metavar="number"),
    make_option(c("-s", "--quant_step"), type="integer", default=5,
                help="step between quantiles [default %default]", metavar="number")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

df <- read.csv(file=opt$file, header=TRUE, sep=",")

qs <- seq(opt$lower_quant/100, opt$upper_quant/100., by=opt$quant_step/100)

df_coeff = NULL

formula = paste(opt$predictand_name, "~", opt$predictor_name)
for (q in qs) {

    this_qr <- rq(formula, data=df, tau = q)
    out = summary(this_qr)
    new_df = as.data.frame(as.table(out$coefficients))
    new_df$quantile <- q

    df_coeff = rbind(df_coeff, new_df)
}

short_fname = tail(strsplit(opt$file, "/")[[1]], 1)
station_name = head(strsplit(short_fname, "_")[[1]], 1)

df_coeff$station_id = station_name

colnames(df_coeff) <- c("coeff_name", "val_type", "value", "quantile", "station_id")
df_coeff <- df_coeff[, c(5, 4, 1, 2, 3)]

write.csv(df_coeff, file=gsub("toR", "QR", opt$file))
