library(dplyr)
library(tidyr)
library(car)
library(ggplot2)
library(ggsignif)
library(ggpubr)


sen_brain_nc_2n.wide <- read.table("../data/1d_2n_sen_bra_onlyN1N2.csv", 
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
  scale_y_continuous(name = "TSE Complexity", limits=c(0, 2.2),
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
  "../data/1d_2n_sen_bra_combined_onlyN1N2.csv", 
  sep=',', header=TRUE, fill=TRUE)
names(sen_brain_nc_2n_joint.wide) <- c('generalist', 'specialist')

sen_brain_nc_2n_joint <- gather(sen_brain_nc_2n_joint.wide, "condition", "neural_complexity")
sen_brain_nc_2n_joint$condition <- as.factor(sen_brain_nc_2n_joint$condition)

ggplot(sen_brain_nc_2n_joint, aes(x=condition, y=neural_complexity)) + 
  geom_boxplot()

leveneTest(sen_brain_nc_2n_joint$neural_complexity, sen_brain_nc_2n_joint$condition) # not significant
n2.joint.model <- t.test(neural_complexity ~ condition, data = sen_brain_nc_2n_joint,
                         paired=FALSE)
n2.joint.model # not significant

my_comparisons = list(c("generalist", "specialist"))

joint.plot <- ggboxplot(sen_brain_nc_2n_joint, x = "condition", y = "neural_complexity") + 
  # stat_compare_means(method="t.test", label.y = 46, size=8) +
  stat_compare_means(comparisons = my_comparisons, label.y = c(2.3),
                     label = "p.signif", size=8) +
  scale_y_continuous(name = "TSE Complexity", limits=c(0, 2.5),
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
  "../data/1d_2n_sen_bra_combined_onlyN1N2_gen_seeds_TSE.csv", 
  sep=',', header=TRUE, fill=TRUE)
tse.individual <- gather(tse.individual.wide, key="seed", 
                         value="neural_complexity", -GEN)

ggplot(tse.individual, aes(y=neural_complexity, group=seed)) + 
  geom_line()


