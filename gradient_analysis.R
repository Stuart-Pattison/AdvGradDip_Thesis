packages <- c('lme4', 'car', 'plotly', 'ggplot2', 'GGally', 'lmerTest', 'lattice', 'MuMIn', 'dplyr', 'effects', 'ggeffects')
lapply(packages, library, character.only=TRUE)


df = read.csv('/home/stuart/Documents/Uni/Graduate Diploma Advanced/Thesis/Data/pmi_worddis.csv', header = TRUE)
df$pro <- as.factor(dp$pro)
df$particle <- as.factor(df$particle) 
df$verb <- as.factor(df$verb)
df$pv <- interaction(df$verb, df$particle)
df$pv <- droplevels(df$pv)

# to remove pronoun cases
df <- subset(df, pro!="True") 
df <- subset(df, select = -c(pro))

# finding mean distance
dp <- df%>%group_by(pv)%>%summarise(wd=mean(wd), pmi = unique(pmi), particle = unique(particle), verb = unique(verb))
dp <- droplevels(dp)

#plotting raw values
hist(dp$pmi, main="")
hist(dp$wd, main="")
barplot(table(dp$particle), main="")
table(dp$pv)
plot(dp$pmi,dp$wd, main="")

#response variable 
dp$wd.lg <- scale(log((dp$wd + 0.01)/(max(dp$wd) +0.01)))
hist(dp$wd.lg)
hist(dp$wd.z <- scale(dp$wd))
plot(dp$wd, dp$wd.lg, main="")

# indep varaible
hist(dp$pmi.z <- scale(dp$pmi))
plot(dp$pmi.z, dp$wd.lg, main="")
plot(dp$pmi.z, dp$wd.z, main="") 

# looking at categorical
dp$particle <- as.factor(dp$particle) 
dp$verb <- as.factor(dp$verb)

# fitting the models
mod.baseline <- lmer(wd.lg ~  1 + (1|particle), data = dp, REML = TRUE)

# linear model
mod.lin.fin <- lmer(wd.lg ~ 1 + pmi.z + (1|particle), data=dp, REML = TRUE)
summary(mod.lin.fin)
qqnorm(residuals(mod.lin.fin)); qqline(residuals(mod.lin.fin)) # looks okay
anova(mod.lin.fin, mod.baseline, test="Chisq")
(exp((AIC(mod.baseline)-AIC(mod.lin.fin))/2)) # still not likely
r.squaredGLMM(mod.lin.fin) # still not a great fit

fixef(mod.lin.fin)
ranef(mod.lin.fin)
Confint(mod.lin.fin)

# quadratic model
mod.pol.fin <- lmer(wd.lg ~ 1 + poly(pmi.z,2) + (1|particle), data=dp, REML = TRUE)
summary(mod.pol.fin)
qqnorm(residuals(mod.pol.fin)); qqline(residuals(mod.pol.fin)) # looks okay
anova(mod.pol.fin, mod.baseline, test="Chisq")
(exp((AIC(mod.baseline)-AIC(mod.pol.fin))/2)) 
r.squaredGLMM(mod.pol.fin)

fixef(mod.pol.fin)
ranef(mod.pol.fin)
Confint(mod.pol.fin)                


#comparing lin and pol
anova(mod.pol.fin, mod.lin.fin, test="Chisq")
(exp((AIC(mod.lin.fin)-AIC(mod.pol.fin))/2)) # much more likely




