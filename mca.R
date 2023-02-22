library(foreign)
library(VIM)
library(missMDA)
library(naniar)
#library(poLCA)
library("FactoMineR")
library("factoextra")
df <- read.csv(file = 'dataframe/df_o_fill.csv')
for (i in 1:55){df[,i] <- as.character(df[,i])}
res.impute <- imputeMCA(df, ncp=50) 
df <- res.impute$completeObs
write.csv(df,"dataframe/MCA_df_ofill.csv", row.names = FALSE)