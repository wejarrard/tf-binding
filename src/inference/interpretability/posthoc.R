library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript posthoc.R <min_seqlet> <receptor_name> <output_dir>")
}

min_seqlet <- as.numeric(args[1])
receptor_name <- args[2]
dir <- args[3]

pwm <- read_csv(paste0(dir, '/lev_pwm.csv'))  %>%
  filter(nchar(sequence) >= min_seqlet)

pos_seqlets <- read_csv(paste0(dir, '/positive_seqlets.csv')) %>%
  filter(nchar(sequence) >= min_seqlet)
neg_seqlets <- read_csv(paste0(dir, '/negative_seqlets.csv')) %>%
  filter(nchar(sequence) >= min_seqlet)

seqlets <- rbind(pos_seqlets, neg_seqlets) %>%
  arrange(desc(abs(attribution)))
write_csv(seqlets, file = paste0(dir, '/all_seqlets.csv'))

(most_common_seqlets <- pos_seqlets %>%
    group_by(sequence) %>%
    summarize(n = n()) %>%
    arrange(desc(n)))
write_csv(most_common_seqlets,
          paste0(dir, '/abundant_candidate_motifs.csv'))
MCS <- unique(slice_max(most_common_seqlets, n, n = 10)$sequence)


tb <- left_join(seqlets, pwm, by = c('example_idx', 'start', 'end'),
                 suffix = c("", "")) %>%
  mutate(common = (sequence %in% MCS))
write_csv(tb, file = paste0(dir, '/seqlets_with_PWM.csv'))

tb <- tb %>%
  arrange(common) %>%
  mutate(orderrank = seq(nrow(tb)))

hist(tb$attribution)
hist(tb$levenshtein_score)
hist(pwm$levenshtein_score)

best_attributions <- pos_seqlets %>%
  slice_max(attribution, prop = 0.05) %>%
  left_join(pwm, by = c('example_idx', 'start', 'end'), suffix = c("", ""))

attr_volc <- ggplot(tb, aes(x = levenshtein_score, y = attribution, order=orderrank)) +
  geom_point(alpha = 1/3, aes(col = common, size = nchar(sequence))) +
  labs(title = "Is PWM similarity related to Attribution Score?",
       x = paste0("Similarity to ", receptor_name, " PWM"),
       y = "Aggregated DeepLift Attribution Score",
       size = "Seqlet Length", color = "Common?") +
  theme_bw()
attr_volc
ggsave(attr_volc, filename = paste0(dir, '/attr_volc.png'),
       width = 1920, height = 1080, units = "px", scale = 2)


pos_volc <- ggplot(filter(tb, attribution > 0),
                   aes(x = levenshtein_score, y = attribution, order = orderrank)) +
  geom_point(alpha = 1/5, aes(col = common, size = nchar(sequence))) +
  labs(title = "Positive-Attribution Seqlets",
       x = paste0("Similarity to ", receptor_name, " PWM"),
       y = "Aggregated DeepLift Attribution Score",
       size = "Seqlet Length", color = "Common?") +
  theme_bw()
pos_volc
ggsave(pos_volc, filename = paste0(dir, '/pos_volc.png'),
       width = 1920, height = 1080, units = "px", scale = 2)


ggplot(best_attributions, aes(x = levenshtein_score, y = attribution)) +
  geom_point(alpha = 1/5) +
  geom_smooth(se = FALSE, method = "lm") +
  theme_bw()




# Overlap Analysis --------------------------------------------------------

best_attributions <- pos_seqlets %>%
  slice_max(attribution, prop = 0.05)
best_seqlets <- unique(best_attributions$sequence)

glimpse(inner_join(best_attributions, pwm,
                   by = c('example_idx', 'start', 'end', 'sequence')))

gaaca <- filter(pos_seqlets, sequence == "GAACA")
for (index in gaaca$example_idx) {
  tmp_tb <- filter(pos_seqlets, example_idx == index)
  if (head(tmp_tb$sequence[[1]] == "GAACA")) {print(index)}
}

com_motifs <- ggplot(slice_max(most_common_seqlets, order_by = n, n = 10),
                     aes(x = reorder(sequence, -n), y = n)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 Most Abundant Seqlets",
       x = "Sequence",
       y = "Count") +
  theme_bw()
ggsave(com_motifs, filename = paste0(dir, '/common_motifs.png'),
       width = 1920, height = 1080, units = "px", scale = 2)

top_attributions <- ggplot(slice_max(best_attributions, attribution, n = 25),
                           aes(x = reorder(sequence, -attribution),
                               y = attribution)) +
  geom_bar(stat = 'identity') +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust = 1)) +
  labs(title = "Highest-Attribution Seqlets",
       x = "Sequence",
       y = "Attribution")
ggsave(top_attributions, filename = paste0(dir, '/top_attrs.png'),
       width = 1920, height = 1080, units = "px", scale = 2)

glimpse(slice_min(tb, attribution, n=1))