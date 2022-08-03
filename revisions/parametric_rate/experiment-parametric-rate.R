setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)

df <- read_csv("experiment-parametric-rate.csv", col_type=cols(n="f", power="f"))
df <- df %>% group_by(n, power) %>%
  summarise(reject_rate=mean(reject)) %>%
  ungroup() %>%
  transform(n = as.numeric(as.character(n))) 

ggplot(df, aes(x=n, y=reject_rate, colour=power)) +
  geom_line() + 
  labs(colour="Rate") +
  scale_x_log10()
