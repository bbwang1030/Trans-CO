args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
seed <- as.integer(args[3])

set.seed(seed)
suppressPackageStartupMessages(library(glmnet))

source(file.path("R_baselines", "TransLasso-functions.R"))

read_mat <- function(path) {
  as.matrix(read.csv(path, header = FALSE))
}

read_vec <- function(path) {
  as.numeric(read.csv(path, header = FALSE)[, 1])
}

X_target <- read_mat(file.path(data_dir, "X_target.csv"))
Y_target <- read_vec(file.path(data_dir, "Y_target.csv"))

K <- as.integer(readLines(file.path(data_dir, "K.txt"))[1])

X_all <- X_target
Y_all <- Y_target
n_vec <- c(nrow(X_target))

for (k in 1:K) {
  Xk <- read_mat(file.path(data_dir, paste0("X_source_", k, ".csv")))
  Yk <- read_vec(file.path(data_dir, paste0("Y_source_", k, ".csv")))

  X_all <- rbind(X_all, Xk)
  Y_all <- c(Y_all, Yk)
  n_vec <- c(n_vec, nrow(Xk))
}

# set.seed(1)
n0 <- nrow(X_target)
I_til <- sort(sample(1:n0, size = max(5, floor(0.2 * n0)), replace = FALSE))

fit <- Trans.lasso(
  X = X_all,
  y = Y_all,
  n.vec = n_vec,
  I.til = I_til,
  l1 = TRUE
)

beta_hat <- as.numeric(fit$beta.hat)

write.table(
  beta_hat,
  file = file.path(out_dir, "translasso_beta.csv"),
  row.names = FALSE,
  col.names = FALSE,
  sep = ","
)
