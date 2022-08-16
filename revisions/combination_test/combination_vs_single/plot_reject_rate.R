library(tidyverse)
use.tikz <- T

# Load data
df.linear <- read_csv("linear/latest.csv", col_type=cols(n="f", method="f", causal_effect="f")) %>% transform(SCM = "Linear")
df.nonlinear <- read_csv("scm/latest.csv", col_type=cols(n="f", method="f", causal_effect="f")) %>% transform(SCM = "Non-linear")
df <- rbind(df.linear, df.nonlinear)

# Take average rejection rates
df <- df %>% group_by(n, method, causal_effect, SCM) %>%
  summarise(reject_rate=mean(reject)) %>%
  ungroup() %>%
  transform(n = as.numeric(as.character(n))) 

# Names for nice plot
method.names <- c("hartung"="Hartung, 1998", "meinshausen"="Meinshausen et al., 2008", "single-test"="Single test", "cct"="Liu and Xie, 2020")
causal_effect.names <- c("0.0" = "No effect present", "0.5" = "Effect present")
scm.names <- c("Linear"="Linear", "Non-linear"="Non-linear")


path = "compare-with-without-testing"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 8, height = 4)}


p <- df %>% 
  subset(!method %in% c("resample+permutation", "resample+permutation+combination")) %>%
  ggplot(aes(x=n, y=reject_rate, colour=method)) +
    geom_line() + 
    labs(colour="Rate") +
    # scale_x_log10() +
    geom_hline(yintercept = 0.05) +
    ylim(0, 1) +
    theme_minimal() + 
    labs(x="$n$", y="Rejection rate") +
    scale_color_brewer(palette = "Dark2", labels=method.names, breaks=names(method.names)) +
    facet_grid(SCM~causal_effect, labeller=as_labeller(c(causal_effect.names, scm.names)))
print(p)

if(use.tikz){
  dev.off()
  print(p)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"))
  
}