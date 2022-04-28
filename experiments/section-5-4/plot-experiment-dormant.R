setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(gridExtra)
library(grid)
library(RColorBrewer)
library(tikzDevice)

use.tikz <- F

grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
  
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[1]] + theme(legend.position = position))$grobs
  legend <- g[[which(sapply(g, function(x) x$name) == "guide-box")]]
  lheight <- sum(legend$height)
  lwidth <- sum(legend$width)
  gl <- lapply(plots, function(x) x + theme(legend.position="none"))
  gl <- c(gl, ncol = ncol, nrow = nrow)
  
  combined <- switch(position,
                     "bottom" = arrangeGrob(do.call(arrangeGrob, gl),
                                            legend,
                                            ncol = 1,
                                            heights = unit.c(unit(1, "npc") - lheight, lheight)),
                     "right" = arrangeGrob(do.call(arrangeGrob, gl),
                                           legend,
                                           ncol = 2,
                                           widths = unit.c(unit(1, "npc") - lwidth, lwidth)))
  
  grid.newpage()
  grid.draw(combined)
  
  # return gtable invisibly
  invisible(combined)
  
}

# Setup tikz
path = "../overleaf-clone/figures/dormant-combined"
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 5.5, height = 2)}


# Load
df <- read_delim("experiment-dormant-nonparametric.csv", delim=",", col_types = cols(n="n", m_choice="f")) %>%
  transform(Replacement = "NO-REPL-reject")

# Set naming of resampling
resampling.names = c(
  "NO-REPL-reject"="$\\Psi_{\\texttt{DRPL}}$"
)
df$Replacement <- factor(df$Replacement, levels=names(resampling.names), labels = resampling.names)

graph.names = c(
  "0"="True graph $\\mathcal{G}$",
  "0.3"="True graph $\\mathcal{H}$"
)
df$Causal_Effect <- factor(df$Causal_Effect, levels=names(graph.names), labels = graph.names)

# Change level ordering to match from binary data loaded below
df$m_choice <-  factor(df$m_choice, levels=rev(levels(df$m_choice)))


df$m_choice <- factor(df$m_choice, levels = c("none", levels(df$m_choice)))

# df$Test = factor(df$method):df$m_choice
# levels(df$Test) <- list(score_based=c("score-based:sqrt", "score-based:heuristic"), 
#                         sqrt="resampling:sqrt", 
#                         heuristic="resampling:heuristic")

# Set confidence level
conf.level = 0.05

p3 <- df %>%
  # subset((method == "resampling") | (m_choice == "heuristic")) %>%
  # ggplot(aes(x=n,y=alpha, colour=Test, fill=Test)) + 
  ggplot(aes(x=n,y=alpha, colour=m_choice, fill=m_choice)) +
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level, fill="confint"), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(size=0.3) +
  geom_line() +
  ggtitle("Non-Gaussian data") +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), size=0.3,alpha=.2, fill="grey70")+
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.3, show.legend = T) +
  labs(y=NULL, x="Number of observations", colour = "Sample size $n$", fill = "Sample size $n$")+
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}, limits=c(0,1)) +
  scale_x_continuous(trans="log10") + 
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name=NULL) +
  facet_wrap(~Causal_Effect) +
  scale_color_brewer(palette = "Dark2", drop=F) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), 
        legend.margin=margin(-5, 0, 0, 0),
        plot.margin = margin(0,5,10,0))


# Load
df <- read_delim("experiment-dormant-continuous.csv", delim=",", col_types = cols(n="n", m_choice="f")) %>%
  transform(Replacement = "NO-REPL-reject")

# Set naming of resampling
resampling.names = c(
  "NO-REPL-reject"="$\\Psi_{\\texttt{DRPL}}$"
)
df$Replacement <- factor(df$Replacement, levels=names(resampling.names), labels = resampling.names)

graph.names = c(
  "0"="True graph $\\mathcal{G}$",
  "0.3"="True graph $\\mathcal{H}$"
)
df$Causal_Effect <- factor(df$Causal_Effect, levels=names(graph.names), labels = graph.names)

# Change level ordering to match from binary data loaded below
df$m_choice <-  factor(df$m_choice, levels=rev(levels(df$m_choice)))

# Set confidence level
conf.level = 0.05
# Make Test variable
df$Test = factor(df$method):df$m_choice
levels(df$Test) <- list(score_based=c("score-based:sqrt", "score-based:heuristic"), 
                     sqrt="resampling:sqrt", 
                     heuristic="resampling:heuristic")




p1 <- df %>%
  subset((method == "resampling") | (m_choice == "heuristic")) %>%
  ggplot(aes(x=n,y=alpha, colour=Test, fill=Test)) + 
  geom_ribbon(aes(colour=NULL,ymin=0,ymax=conf.level, fill="confint"), alpha=0.8, fill="grey70", show.legend = F) +
  geom_point(size=0.3) +
  geom_line() +
  ggtitle("Gaussian data") +
  geom_ribbon(aes(ymax=Upper, ymin=Lower), size=0.3,alpha=.2, fill="grey70")+
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.3, show.legend = T) +
  labs(y=NULL, x="Number of observations", colour = "Sample size $n$", fill = "Sample size $n$")+
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}) +
  scale_x_continuous(trans="log10") + 
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name=NULL) +
  facet_wrap(~Causal_Effect) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5), legend.margin=margin(-5, 0, 0, 0),
        plot.margin = margin(0,2,10,0))

# Load
df <- read_delim('experiment-dormant-binary.csv', delim = ",")

graph.names = c(
  "null"="True graph $\\mathcal{G}$",
  "alternative"="True graph $\\mathcal{H}$"
)
df$Hypothesis <- factor(df$Hypothesis, levels=names(graph.names), labels = graph.names)

Test.names = c(Score_based = 'Score-based',
               Resampling1 = 'Resampling ($m=\\sqrt{n}$)',
               Resampling2 = 'Resampling: (target heuristic $m$)')
df$Test <- factor(df$Test, levels=names(Test.names), labels=Test.names)



# Set confidence level
conf.level = 0.05

p2 <- df %>%
  ggplot(aes(x = n, y = alpha, colour = Test, fill = Test)) +
  geom_ribbon(aes(colour = NULL, ymin = 0, ymax = conf.level, fill = "confint"), alpha = 0.8, fill = "grey70", show.legend = F) +
  geom_point(size = 0.3) +
  geom_line() +
  ggtitle("Binary data") +
  geom_ribbon(aes(ymax = Upper, ymin = Lower), size = 0.3, alpha = .2, fill = "grey70", show.legend = F) +
  geom_hline(aes(lty="5\\% level", yintercept=conf.level), size=0.3, show.legend = T) +
  labs(y = "Rejection rate", x = "Number of observations") +
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}, 
                     name = "Rejection Rate"
  ) +
  scale_x_continuous(trans="log10") + 
  scale_linetype_manual(values = c("22"), breaks = "5\\% level", name = NULL) +
  theme_bw() +
  scale_color_brewer(palette = "Dark2") +
  facet_wrap(~Hypothesis) +
  theme(plot.title = element_text(hjust = 0.5), legend.margin=margin(-5, 0, 0, 0),
        plot.margin = margin(0,2,10,-10))

grid_arrange_shared_legend(p2, p1, p3)


if(use.tikz){
  dev.off()
  grid_arrange_shared_legend(p2, p1, p3)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
}
