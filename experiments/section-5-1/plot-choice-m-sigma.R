setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(gridExtra)
library(grid)
library(tikzDevice)
library(RColorBrewer)

grid_arrange_shared_legend <- function(..., ncol = length(list(...)), nrow = 1, position = c("bottom", "right")) {
  
  plots <- list(...)
  position <- match.arg(position)
  g <- ggplotGrob(plots[[2]] + theme(legend.position = position))$grobs
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



use.tikz <- F
caus_eff = "0.4"

# Reject names
reject.names = c("TRUE" = "$p < 0.05$", "FALSE" = "$p \\geq 0.05$")

# Load
df <- read_delim("choice-m.csv", delim=",", col_types = cols(m="n", n = "f", causal_effect="f")) %>%
  subset(!is.na(pval)) %>%
  transform(Rejected = factor(pval < 0.05, levels=names(reject.names), labels=reject.names))

# Set levels for n
n.names = paste0("$n=", levels(df$n), "$")
names(n.names) <- levels(df$n)
df$n <- factor(df$n, levels=names(n.names), labels=n.names)

# Set names of causal effect
effect_names <- c("0.0" = "Null", "0.4" = "Alt.")
df$causal_effect <- factor(df$causal_effect, levels=names(effect_names), labels = effect_names)


# Compute quantile and first break
reps <- 100; K <- 10000; X <- matrix(runif(reps*K),reps,K);
q = quantile(apply(X, 2, mean), probs = 0.05)
df %>% group_by(m, causal_effect) %>%
  summarise(meanp = mean(pval),
            meanhyp = mean(hyp_test),
            is.valid = mean(pval) > q) %>%
  group_by(causal_effect) %>%
  mutate(all.valid = (cumprod(is.valid) == 1)) %>%
  subset(all.valid) %>%
  filter(m == max(m)) -> max_points

###
df_hyp <- read_delim("choice-m-ci.csv", delim=",", col_types = cols(idx="f", m="n", n = "f", causal_effect="f"))
df_hyp$causal_effect <- factor(df_hyp$causal_effect, levels=names(effect_names), labels = effect_names)

# Get conservative m-choices chosen by finite-sample model
df_m <- read_delim("choice-m-m-tuning.csv", delim=",", col_types = cols(m="n", causal_effect="f"))
df_m$causal_effect <- factor(df_m$causal_effect, levels=names(effect_names), labels = effect_names)

df_m <- df_m %>% group_by(causal_effect) %>% summarise(m = mean(m))
df_m_tmp <- df %>% group_by(m, causal_effect) %>% 
  summarise(meanp = mean(pval), meanhyp = mean(hyp_test)) 

df_m_tmp_1 = subset(df_m_tmp, causal_effect==df_m$causal_effect[1])
df_m_tmp_2 = subset(df_m_tmp, causal_effect==df_m$causal_effect[2])


df_m <- cbind(df_m, rbind(df_m_tmp_1[which.min(abs(df_m_tmp_1$m - df_m$m[1])),c("meanp","meanhyp")],
                          df_m_tmp_2[which.min(abs(df_m_tmp_2$m - df_m$m[2])),c("meanp","meanhyp")]))


# Setup tikz
path = "choice-m-sigma"
# if(use.tikz){tikz(file=paste0(path, ".tex"), width = 3, height = 2)}
if(use.tikz){tikz(file=paste0(path, ".tex"), width = 5.5, height = 2)}


p1 <- df %>%
  ggplot(aes(x=m, y=pval, group=causal_effect, colour=causal_effect)) +
  geom_point(data = subset(df, rep<=30), alpha=0.3, size=0.5) +
  stat_summary(fun=mean, geom="line")+
  labs(x="$m$", y="Resampling validity", lty=NULL, colour=NULL, shape=NULL) +
  geom_hline(aes(yintercept=(q), lty="Uniform quantile")) +
  geom_hline(aes(yintercept=q, lty="$5$\\% level"), alpha=0) +
  geom_vline(data=data.frame(n = levels(df$n), m=c(sqrt(c(10000)))), mapping=aes(xintercept=m, lty="$m=\\sqrt{n}$ or $\\sigma^2 = 2(\\sigma_{\\epsilon_Z}^2 - \\sigma_{X}^2)$"), show.legend=F, size=0.6) +
  scale_linetype_manual(values=c("44", "13", "4121"), breaks=c("$m=\\sqrt{n}$ or $\\sigma^2 = 2(\\sigma_{\\epsilon_Z}^2 - \\sigma_{X}^2)$", "$5$\\% level", "Uniform quantile")) +
  guides(colour = guide_legend(override.aes = list(alpha = 1))) +
  geom_point(aes(y=meanp), data=max_points, size=4, shape=1, stroke=1.1, show.legend=F) +
  # geom_point(aes(y=meanp), data=df_m, size=4, shape=2, stroke=1.1, show.legend = F) +
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}, limits = c(0, 1)) +
  scale_x_continuous(trans="log10") +
  theme_bw() +
  guides(lty=guide_legend(nrow=1,byrow=F)) +
  scale_color_brewer(palette = "Dark2")+
  theme(
    plot.margin = margin(0,5,0,5),
    legend.spacing.x = unit(0.1, 'cm'),
    legend.spacing.y = unit(-0.3, "cm")
  )

p2 <- df %>%
  ggplot(aes(x=m, y=hyp_test, colour=causal_effect)) + 
  stat_summary(fun=mean, geom="line") + 
  geom_hline(aes(yintercept=0.05, lty="5pct-level")) + 
  geom_vline(data=data.frame(n = levels(df$n), m=c(sqrt(c(10000)))), mapping=aes(xintercept=m), lty="44", show.legend=F, size=0.6) +
  scale_x_continuous(trans="log10") +
  geom_point(aes(y=meanhyp), shape=1, data=max_points, stroke=1.1, size=4) +
  geom_point(aes(y=meanhyp), data=df_m, size=4, shape=2, stroke=1.1, show.legend = F) +
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}, limits = c(0, 1)) +
  stat_summary(data=df_hyp, geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.95), size=0.00, alpha=0.1, show.legend=F)+
  scale_linetype_manual(values=c("13"), breaks=c("5pct-level")) +
  labs(x="$m$", y="Target hypothesis $p$-value") +
  scale_color_brewer(palette = "Dark2")+
  theme_bw() +
  theme(
    plot.margin = margin(0,5,0,5),
  )

#####
# Load
df.a3 <- read_delim("choice-sigma.csv", delim=",", col_types=cols(CausalEffect="f", n="f"))

# Set effect names
effect_names <- c("0.0" = "Null", "0.4" = "Alt.")
df.a3$CausalEffect <- factor(df.a3$CausalEffect, levels=names(effect_names), labels = effect_names)

# Set naming of resampling
resampling.names = c("True"="$\\Psi_{\\texttt{REPL}}$",
                     "False"="$\\Psi_{\\texttt{NO-REPL}}$",
                     "NO-REPL-gibbs"="$\\Psi_{\\texttt{GIBBS}}$",
                     "REPL-reject"="$\\Psi_{\\texttt{DRPL-REPL}}$",
                     "NO-REPL-reject"="$\\Psi_{\\texttt{DRPL}}$"
)
df.a3$Replacement <- factor(df.a3$Replacement, levels=names(resampling.names), labels = resampling.names)


# Set confidence level
conf.level = 0.05
sigma_X = 1
sigma_eps = 2
theo.threshold <- sqrt(2*(sigma_eps**2 - sigma_X**2))



p3 <- df.a3 %>%
  ggplot(aes(x=Scale,y=alpha,colour=CausalEffect)) + 
  geom_point(size=0.3, show.legend = F) +
  geom_line(show.legend = F) +
  geom_hline(aes(yintercept=conf.level), size=0.3, show.legend = F, lty="13") +
  geom_vline(aes(xintercept=theo.threshold), lty="44", show.legend = F, size=0.6) + 
  labs(y="Target hypothesis $p$-value", x="Target sd, $\\sigma$", colour=NULL)+
  scale_x_continuous(trans="log10") +
  scale_y_continuous(labels = function(z){paste0(100*z, "\\%")}, limits = c(0, 1)) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw() + 
  theme(
    plot.margin = margin(0,0,0,5),
  ) +
  guides(color=guide_legend(order=1))

grid_arrange_shared_legend(p2, p1, p3)

if(use.tikz){
  dev.off()
  grid_arrange_shared_legend(p2, p1, p3)
  lines <- readLines(con=paste0(path, ".tex"))
  lines <- lines[-which(grepl("\\path\\[clip\\]*", lines,perl=F))]
  lines <- lines[-which(grepl("\\path\\[use as bounding box*", lines,perl=F))]
  writeLines(lines,con=paste0(path, ".tex"))
  ggsave(paste0(path, ".pdf"))
  
}
