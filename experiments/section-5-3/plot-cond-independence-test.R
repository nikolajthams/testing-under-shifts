setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tikzDevice)
library(RColorBrewer)
library(ggh4x)

use.tikz <- F

# Load
df <- read_delim("cond-independence-test.csv", delim=",", col_types = cols(n="f", EffectOrder="f"))
df$Test[df$Test == "LinReg"] = "CorTest $+ \\Psi_{\\texttt{DRPL}}$"
df$Test[df$Test == "HSIC"] = "HSIC $+ \\Psi_{\\texttt{DRPL}}$"
df$Test[df$Test == "HSICfit"] = "HSICfit $+ \\Psi_{\\texttt{DRPL}}$"

# Order names
order_names <- c("1" = "Linear", "2" = "Quadratic")
df$EffectOrder <- factor(df$EffectOrder, levels=names(order_names), labels = order_names)

# Factor for plotting conditional vs marginal tests
test.type_names <- c("marg" = "Resampling and marginal test", "cond" = "Conditional test")
df$TestType <- factor(ifelse(df$Test %in% c("GCM", "KCI"), "cond", "marg"), levels=names(test.type_names), labels=test.type_names)

# Set confidence level
conf.level = 0.05

# Setup tikz
path = "cond-independence-test"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 6, height = 1.75)}

p <- df %>%
  ggplot(aes(x=CausalEffect,y=alpha, colour=EffectOrder, fill=EffectOrder)) + 
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(size=0.3) +
  geom_line() +
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.1,  show.legend = T) +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), fill="grey70", size=0.3,alpha=.2)+
  labs(y="Rejection rate", x="Strength of direct effect", fill = "Direct effect", colour="Direct effect")+
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}) +
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name=NULL) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw() +
  facet_nested(~TestType + Test)

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