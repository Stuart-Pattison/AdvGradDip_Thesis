packages <- c('lme4', 'car', 'plotly', 'ggplot2', 'GGally', 'lmerTest', 'lattice', 'MuMIn', 'dplyr', 'effects', 'ggeffects', 'DHARMa', 'ggpubr', 'ggthemes')
lapply(packages, library, character.only=TRUE)


df = read.csv('/home/stuart/Documents/Uni/Graduate Diploma Advanced/Thesis/Data/pmi_worddis.csv', header = TRUE)
df$pro <- as.factor(df$pro)
df$particle <- as.factor(df$particle) 
df$verb <- as.factor(df$verb)
df$pv <- interaction(df$verb, df$particle)
df$pv <- droplevels(df$pv)

# to remove pronoun cases
df <- subset(df, pro!="True") 
df <- subset(df, select = -c(pro))

# to turn into binary split data
df$split <- factor(ifelse(df$wd==0, 0, 1))
df$pmi.z <- scale(df$pmi)
summary(df)
plot(df$pmi.z, as.numeric(df$split))

# linear binomial model
mod.bi.lin <- glmer(split ~ 1 + pmi.z + (1|particle) + (1|verb), family=binomial, data=df)
summary(mod.bi.lin)
sim.bi.lin <- simulateResiduals(fittedModel = mod.bi.lin)
plotResiduals(sim.bi.lin, df$pmi.z)
plotResiduals(sim.bi.lin)
plot(sim.bi.lin)
testDispersion(sim.bi.lin)
testZeroInflation(sim.bi.lin)

# quadratic binomial model
mod.bi.pol <- glmer(split ~ 1 + poly(pmi.z,2) + (1|particle) + (1|verb), family=binomial, data=df)
summary(mod.bi.pol)
sim.bi.pol <- simulateResiduals(fittedModel = mod.bi.pol)
plotResiduals(sim.bi.pol, df$pmi.z)
plotResiduals(sim.bi.pol)
plot(sim.bi.pol)
testDispersion(sim.bi.pol)
testZeroInflation(sim.bi.pol)

mod.bi.baseline <- glmer(split ~ 1 + (1|particle) + (1|verb), family=binomial, data=df)

anova(mod.bi.lin, mod.bi.baseline, test="Chisq")
anova(mod.bi.pol, mod.bi.baseline, test="Chisq")

(exp((AIC(mod.bi.baseline)-AIC(mod.bi.lin))/2)) # much more likely
(exp((AIC(mod.bi.baseline)-AIC(mod.bi.pol))/2)) # more  likely
(exp((AIC(mod.bi.lin)-AIC(mod.bi.pol))/2)) # much more likely

summary(mod.bi.pol)

