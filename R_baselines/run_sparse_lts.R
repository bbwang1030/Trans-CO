options(repos = c(CRAN = "https://cloud.r-project.org"))

args <- commandArgs(trailingOnly = TRUE)

data_dir <- args[1]
out_dir <- args[2]
seed <- as.integer(args[3])


set.seed(seed)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("robustHD", quietly = TRUE)) {
  stop("Package 'robustHD' is not installed. Please run install.packages('robustHD').")
}

suppressPackageStartupMessages(library(robustHD))

read_mat <- function(path) {
  if (!file.exists(path)) {
    stop(paste("Missing file:", path))
  }
  as.matrix(read.csv(path, header = FALSE))
}

read_vec <- function(path) {
  if (!file.exists(path)) {
    stop(paste("Missing file:", path))
  }
  as.numeric(read.csv(path, header = FALSE)[, 1])
}

X_target <- read_mat(file.path(data_dir, "X_target.csv"))
Y_target <- read_vec(file.path(data_dir, "Y_target.csv"))

cat("Sparse LTS: dim(X_target) =", dim(X_target), "\n")
cat("Sparse LTS: length(Y_target) =", length(Y_target), "\n")

# set.seed(1)

lambda_grid <- seq(0.001, 1, length.out = 20)


fit <- sparseLTS(
  x = X_target,
  y = Y_target,
  lambda = lambda_grid,
  mode = "fraction",
  alpha = 0.85,
  intercept = FALSE,
  crit = "BIC",
  seed = seed,
  ncores = NA
)

beta_hat <- as.numeric(coef(fit))

if (length(beta_hat) != ncol(X_target)) {
  if (length(beta_hat) == ncol(X_target) + 1) {
    beta_hat <- beta_hat[-1]
  } else {
    cat("coef length =", length(beta_hat), "\n")
    cat("p =", ncol(X_target), "\n")
    stop("Sparse LTS coefficient length does not match p.")
  }
}

write.table(
  beta_hat,
  file = file.path(out_dir, "sparse_lts_beta.csv"),
  row.names = FALSE,
  col.names = FALSE,
  sep = ","
)

# Extract weights corresponding to selected model
wt <- weights(fit)

if (is.matrix(wt)) {
  stop(
    paste(
      "weights(fit) returned weights for multiple lambda values.",
      "Do not automatically take the last column;",
      "extract the BIC-selected model instead."
    )
  )
}

wt <- as.numeric(wt)

if (length(wt) != nrow(X_target)) {
  stop("Sparse LTS weights length does not match target sample size.")
}

outlier_hat <- as.numeric(wt == 0)

write.table(
  outlier_hat,
  file = file.path(out_dir, "sparse_lts_outlier_hat.csv"),
  row.names = FALSE,
  col.names = FALSE,
  sep = ","
)

cat("Sparse LTS finished successfully.\n")
