library(foreign)
library(VIM)
library(missMDA)
library(naniar)
library('missForest')
library('randomForest')
library("FactoMineR")
library("factoextra")
df <- read.csv(file = '/home/elina/Documents/thesis/df_o_fill.csv')
for (i in 1:55){df[,i] <- as.factor(df[,i])}
df[df == 'NaN'] <- NA
df_im <- missForest(df)
df <- df_im$ximp
df <- type.convert(df)
summary(df)
write.csv(df,"/home/elina/Documents/thesis/missForest_df_ofill.csv", row.names = FALSE)