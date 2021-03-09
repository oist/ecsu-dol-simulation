library(dplyr)
library(tidyr)
library(car)
library(ggplot2)
library(ggsignif)
library(ggpubr)


brain_nc_4n.wide <- read.table("../data/4n_bra.csv", sep=',', header=TRUE,
                          fill=TRUE)
brain_nc_4n.wide <- select(brain_nc_4n.wide, -X)
brain_nc_4n <- gather(brain_nc_4n.wide, "condition", "neural_complexity")

ggplot(brain_nc_4n, aes(x=condition, y=neural_complexity)) + 
  geom_boxplot()


z_scores <- as.data.frame(scale(brain_nc_4n.wide))
no_outliers <- z_scores[!rowSums(z_scores>3), ]
no_outliers <- gather(no_outliers, "condition", "neural_complexity")
ggplot(no_outliers, aes(x=condition, y=neural_complexity)) + 
  geom_boxplot()


leveneTest(no_outliers$neural_complexity, no_outliers$condition) # significant
n4.model <- aov(neural_complexity ~ condition, data = no_outliers)
summary(n4.model) # not significant
# pairwise.t.test(synchrony$distance, synchrony$condition, paired = FALSE, 
#                p.adjust.method = "bonferroni")  # significant only for cond4 vs rest



sen_brain_nc_2n.wide <- read.table("../data/2n_sen_bra.csv", sep=',', header=TRUE,
                               fill=TRUE)
sen_brain_nc_2n.wide <- select(sen_brain_nc_2n.wide, -X)
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

ggboxplot(sen_brain_nc_2n, x = "condition", y = "neural_complexity") + 
  stat_compare_means(method = "anova", label.y=3.5, size=8) +
  stat_compare_means(comparisons = my_comparisons, label.y = c(3.1, 2.9),
                     label = "p.signif", method = "t.test", p.adj = "bonferroni",
                     size=8) +
  scale_y_continuous(name = "TSE Complexity") +
  scale_x_discrete(name = "Condition") + 
  theme(axis.text = element_text(size = 16),
        axis.title=element_text(size=20), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))


sen_brain_nc_2n_joint.wide <- read.table("../data/2n_sen_bra_combined.csv", sep=',', header=TRUE,
                                   fill=TRUE)
sen_brain_nc_2n_joint.wide <- select(sen_brain_nc_2n_joint.wide, -X)
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

ggboxplot(sen_brain_nc_2n_joint, x = "condition", y = "neural_complexity") + 
  stat_compare_means(method="t.test", label.y = 46, size=8) +
  stat_compare_means(comparisons = my_comparisons, label.y = c(43),
                     label = "p.signif", size=8) +
  scale_y_continuous(name = "TSE Complexity") +
  scale_x_discrete(name = "Condition") + 
  theme(axis.text = element_text(size = 16),
        axis.title=element_text(size=20), panel.background = element_blank(),
        axis.line = element_line(colour = "grey"),
        axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0)))
