library(dplyr)
library(tidyr)
library(car)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(pastecs)
library(compute.es)


sen_brain_nc_2n.wide <- read.table(
  "../analysis_alife21/1d_2n_box_TSE_sen_bra_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
names(sen_brain_nc_2n.wide) <- c('individual', 'generalist', 'specialist.left', 'specialist.right')
sen_brain_nc_2n.wide <- mutate(sen_brain_nc_2n.wide, 
                               specialist = rowMeans(select(sen_brain_nc_2n.wide, 
                                                            starts_with('specialist')), 
                                                     na.rm = TRUE))
sen_brain_nc_2n.wide <- select(sen_brain_nc_2n.wide, -c('specialist.left', 'specialist.right'))

sen_brain_nc_2n <- gather(sen_brain_nc_2n.wide, "condition", "neural_complexity")
sen_brain_nc_2n$condition <- as.factor(sen_brain_nc_2n$condition)
sen_brain_nc_2n$condition <- relevel(sen_brain_nc_2n$condition, 'individual')


ggplot(sen_brain_nc_2n, aes(x=condition, y=neural_complexity)) + 
  geom_boxplot()

leveneTest(sen_brain_nc_2n$neural_complexity, sen_brain_nc_2n$condition) # not significant
n2.model <- aov(neural_complexity ~ condition, data = sen_brain_nc_2n)
summary(n2.model) # significant
pairwise.t.test(sen_brain_nc_2n$neural_complexity, sen_brain_nc_2n$condition, paired = FALSE, 
                p.adjust.method = "bonferroni")  # significant for generalist vs the others

descr <- by(sen_brain_nc_2n$neural_complexity, sen_brain_nc_2n$condition, stat.desc)
mes(descr$individual["mean"], descr$generalist["mean"], 
    descr$individual["std.dev"], descr$generalist["std.dev"], 16, 13)
mes(descr$specialist["mean"], descr$generalist["mean"], descr$specialist["std.dev"], 
    descr$generalist["std.dev"], 19, 13)


# ggplot(sen_brain_nc_2n, aes(x=condition, y=neural_complexity)) + 
#   geom_boxplot() + geom_signif(comparisons = list(c("individual", "generalist")), 
#                                map_signif_level=TRUE)

my_comparisons = list( c("individual", "generalist"), 
                       c("generalist", "specialist"))

ind.plot <- ggboxplot(sen_brain_nc_2n, x = "condition", y = "neural_complexity") + 
  # stat_compare_means(method = "anova", label.y=2.3, size=8) +
  stat_compare_means(comparisons = my_comparisons, label.y = c(1.7, 1.5),
                     label = "p.signif", method = "t.test", p.adj = "bonferroni",
                     size=8) +
  scale_y_continuous(name = "C(X)", limits=c(0, 2.2),
                     breaks=c(0.0, 0.5, 1.0, 1.5, 2.0)) +
  scale_x_discrete(name = "Condition") + 
  theme(axis.text = element_text(size = 14),
        axis.title=element_text(size=18), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))

ggsave("../plots/boxplot_individual.eps", ind.plot, 
       width=18, height=10, units="cm")


sen_brain_nc_2n_joint.wide <- read.table(
  "../analysis_alife21/1d_2n_box_TSE_sen_bra_combined_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
names(sen_brain_nc_2n_joint.wide) <- c('generalist', 'specialist')

sen_brain_nc_2n_joint <- gather(sen_brain_nc_2n_joint.wide, "condition", "neural_complexity")
sen_brain_nc_2n_joint$condition <- as.factor(sen_brain_nc_2n_joint$condition)

ggplot(sen_brain_nc_2n_joint, aes(x=condition, y=neural_complexity)) + 
  geom_boxplot()

leveneTest(sen_brain_nc_2n_joint$neural_complexity, sen_brain_nc_2n_joint$condition) # not significant
n2.joint.model <- t.test(neural_complexity ~ condition, data = sen_brain_nc_2n_joint,
                         paired=FALSE)
n2.joint.model # significant

t <- n2.joint.model$statistic[[1]]
df <- n2.joint.model$parameter[[1]]
eff.size <- sqrt(t^2/(t^2+df))

my_comparisons = list(c("generalist", "specialist"))

joint.plot <- ggboxplot(sen_brain_nc_2n_joint, x = "condition", y = "neural_complexity") + 
  # stat_compare_means(method="t.test", label.y = 46, size=8) +
  stat_compare_means(comparisons = my_comparisons, label.y = c(2.3),
                     label = "p.signif", size=8) +
  scale_y_continuous(name = "C(X)", limits=c(0, 2.5),
                     breaks=c(0.0, 0.5, 1.0, 1.5, 2.0)) +
  scale_x_discrete(name = "Condition") + 
  theme(axis.text = element_text(size = 14),
        axis.title=element_text(size=18), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))

ggsave("../plots/boxplot_joint.eps", joint.plot, 
       width=18, height=10, units="cm")


tse.individual.wide <- read.table(
  "../analysis_alife21/1d_2n_gen_seeds_TSE_individuals_sen_bra_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.individual.wide$condition <- rep("individual", 11)
tse.individual <- gather(tse.individual.wide, key="seed", 
                         value="neural_complexity", -GEN, -condition)
tse.individual$seed <- paste("i", tse.individual$seed, sep="_")

tse.generalist.wide <- read.table(
  "../analysis_alife21/1d_2n_gen_seeds_TSE_generalists_sen_bra_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.generalist.wide$condition <- rep("generalist", 11)
tse.generalist <- gather(tse.generalist.wide, key="seed", 
                         value="neural_complexity", -GEN, -condition)
tse.generalist$seed <- paste("g", tse.generalist$seed, sep="_")


tse.specialist.wide <- read.table(
  "../analysis_alife21/1d_2n_gen_seeds_TSE_specialists_sen_bra_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.specialist.wide$condition <- rep("specialist", 11)
tse.specialist <- gather(tse.specialist.wide, key="seed", 
                         value="neural_complexity", -GEN, -condition)
tse.specialist$seed <- paste("s", tse.specialist$seed, sep="_")


tse.individual.gen <- rbind(tse.individual, tse.generalist, tse.specialist)
tse.individual.gen$condition <- factor(tse.individual.gen$condition)
tse.individual.gen$condition <- relevel(tse.individual.gen$condition, 'individual')

gd <- tse.individual.gen %>% 
  group_by(condition, GEN) %>% 
  summarise(neural_complexity = mean(neural_complexity))

tse.generations <- ggplot(tse.individual.gen, 
                          aes(x=GEN, y=neural_complexity, color=condition)) + 
  geom_line(aes(group=seed), alpha=0.2) +
  geom_line(data=gd, alpha=1, size=0.8) +
  scale_y_continuous(name = "C(X)") +
  scale_x_continuous(name = "Generations") + 
  theme(axis.text = element_text(size = 14),
        axis.title=element_text(size=18), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)),
        legend.title = element_text(size=14),
        legend.text = element_text(size=12)) +
  scale_color_discrete(name = "Condition")

ggsave("../plots/tse_over_generations.pdf", tse.generations, 
       width=18, height=10, units="cm")



tse.generalist.combined.wide <- read.table(
  "../analysis_alife21/1d_2n_gen_seeds_TSE_generalists_sen_bra_combined_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.generalist.combined.wide$condition <- rep("generalist", 11)
tse.generalist.combined <- gather(tse.generalist.combined.wide, key="seed", 
                         value="neural_complexity", -GEN, -condition)
tse.generalist.combined$seed <- paste("g", tse.generalist.combined$seed, sep="_")


tse.specialist.combined.wide <- read.table(
  "../analysis_alife21/1d_2n_gen_seeds_TSE_specialists_sen_bra_combined_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.specialist.combined.wide$condition <- rep("specialist", 11)
tse.specialist.combined <- gather(tse.specialist.combined.wide, key="seed", 
                         value="neural_complexity", -GEN, -condition)
tse.specialist.combined$seed <- paste("s", tse.specialist.combined$seed, sep="_")


tse.combined.gen <- rbind(tse.generalist.combined, tse.specialist.combined)

gd.combined <- tse.combined.gen %>% 
  group_by(condition, GEN) %>% 
  summarise(neural_complexity = mean(neural_complexity))


ggplot(tse.combined.gen, aes(x=GEN, y=neural_complexity, color=condition)) + 
  geom_line(aes(group=seed), alpha=0.2) +
  geom_line(data=gd.combined, alpha=1, size=0.8) +
  scale_y_continuous(name = "C(X)") +
  scale_x_continuous(name = "Generations") + 
  theme(axis.text = element_text(size = 14),
        axis.title=element_text(size=18), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))

