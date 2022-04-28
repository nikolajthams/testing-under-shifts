setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(RColorBrewer)
library(tikzDevice)

use.tikz <- T

# Load
df <- read_delim("compare-to-ipw.csv", delim=",", col_types = cols(Test="f", n="f")) %>%
  transform(level = addNA(factor(RejectRate < 0.05)))

# levels(df$level) <- c(F, T, "Resample size too small to test")


# Setup tikz
path = "compare-to-ipw"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 3, height = 1.5)}

p <- df %>%
  ggplot(aes(x=TargetMean, y=RejectRate, colour=Test)) +
  geom_line() + 
  geom_ribbon(aes(ymax=Upper, ymin=Lower), fill="grey70", size=0.3,alpha=0.1)+
  geom_hline(aes(yintercept=0.05))+
  scale_x_continuous(breaks=(c(1:6)))+
  labs(x="Target mean, $\\mu$", y="Rejection rate") +
  scale_color_brewer(palette = "Dark2") +
  theme_bw()
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