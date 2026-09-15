options(repos = c(CRAN = "https://cloud.r-project.org"))

args0 <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- normalizePath(sub(file_arg, "", args0[grep(file_arg, args0)]))
script_dir <- dirname(script_path)

args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_dir <- args[2]
seed <- as.integer(args[3])

set.seed(seed)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("script_dir =", script_dir, "\n")
cat("data_dir =", data_dir, "\n")
cat("out_dir =", out_dir, "\n")

required_pkgs <- c("glmnet", "Rcpp", "MASS", "Matrix")

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0(
      "Package '", pkg, "' is not installed. ",
      "Please install it first using install.packages('", pkg, "')."
    ))
  }
}

suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(Rcpp))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(Matrix))

transptlr_dir <- file.path(script_dir, "TransPtLR")
r_file <- file.path(transptlr_dir, "Transfer learning.R")
cpp_file <- file.path(transptlr_dir, "updatebeta.cpp")

if (!file.exists(r_file)) {
  stop(paste("Cannot find Transfer learning.R at:", r_file))
}

if (!file.exists(cpp_file)) {
  stop(paste("Cannot find updatebeta.cpp at:", cpp_file))
}

cat("TransPtLR dir =", transptlr_dir, "\n")

# Source the author's code in an isolated environment.
# This avoids pollution from variables such as K, targetdata, sourcedatas, betaTL, etc.
old_wd <- getwd()
setwd(transptlr_dir)

trans_env <- new.env(parent = globalenv())

# Compile updatebeta.cpp into trans_env
Rcpp::sourceCpp("updatebeta.cpp", env = trans_env)

# Parse Transfer learning.R but only evaluate top-level function definitions.
# This avoids running the author's simulation code at the bottom of the file.
exprs <- parse(file = "Transfer learning.R")

is_function_assignment <- function(expr) {
  if (!is.call(expr)) {
    return(FALSE)
  }

  lhs_op <- expr[[1]]

  is_assign <- identical(lhs_op, as.name("<-")) ||
    identical(lhs_op, as.name("=")) ||
    identical(lhs_op, as.name("assign"))

  if (!is_assign) {
    return(FALSE)
  }

  # For ordinary assignments: name <- function(...)
  if (length(expr) >= 3) {
    rhs <- expr[[3]]

    if (is.call(rhs) && identical(rhs[[1]], as.name("function"))) {
      return(TRUE)
    }
  }

  return(FALSE)
}

loaded_functions <- character(0)

for (ii in seq_along(exprs)) {
  expr <- exprs[[ii]]

  if (is_function_assignment(expr)) {
    eval(expr, envir = trans_env)

    fname <- tryCatch(
      as.character(expr[[2]]),
      error = function(e) paste0("expr_", ii)
    )

    loaded_functions <- c(loaded_functions, fname)
  }
}

setwd(old_wd)

cat("Loaded function definitions from Transfer learning.R:\n")
print(loaded_functions)

cat("Objects inside trans_env after function-only loading:\n")
print(ls(trans_env))


if (!exists("PLR", envir = trans_env, inherits = FALSE)) {
  stop("Cannot find PLR() in Transfer learning.R.")
}

if (!exists("TL", envir = trans_env, inherits = FALSE)) {
  stop("Cannot find TL() in Transfer learning.R.")
}

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

K_file <- file.path(data_dir, "K.txt")

if (!file.exists(K_file)) {
  stop(paste("Missing K.txt at:", K_file))
}

K_data <- as.integer(readLines(K_file)[1])

if (is.na(K_data) || K_data <= 0) {
  stop("K.txt must contain a positive integer.")
}

cat("K_data from K.txt =", K_data, "\n")

X_source <- vector("list", K_data)
Y_source <- vector("list", K_data)

for (kk in seq_len(K_data)) {
  x_file <- file.path(data_dir, paste0("X_source_", kk, ".csv"))
  y_file <- file.path(data_dir, paste0("Y_source_", kk, ".csv"))

  if (!file.exists(x_file)) {
    stop(paste("Missing file:", x_file))
  }

  if (!file.exists(y_file)) {
    stop(paste("Missing file:", y_file))
  }

  X_source[[kk]] <- read_mat(x_file)
  Y_source[[kk]] <- read_vec(y_file)

  cat(
    "loaded source", kk,
    "dim =", dim(X_source[[kk]]),
    "length y =", length(Y_source[[kk]]),
    "\n"
  )
}

cat("dim(X_target) =", dim(X_target), "\n")
cat("length(Y_target) =", length(Y_target), "\n")
cat("length(X_source) =", length(X_source), "\n")
cat("length(Y_source) =", length(Y_source), "\n")

targetdata_user <- list(
  n = nrow(X_target),
  x = X_target,
  y = Y_target,
  X = X_target,
  Y = Y_target
)

sourcedatas_user <- vector("list", K_data)

for (kk in seq_len(K_data)) {
  sourcedatas_user[[kk]] <- list(
    n = nrow(X_source[[kk]]),
    x = X_source[[kk]],
    y = Y_source[[kk]],
    X = X_source[[kk]],
    Y = Y_source[[kk]]
  )
}

cat("target n =", targetdata_user$n, "\n")
cat("number of source datasets =", length(sourcedatas_user), "\n")

method_user <- "t"

lambda_user <- seq(0.001, 1, length.out = 20)
lambda_user <- as.numeric(lambda_user)
lambda_user <- lambda_user[is.finite(lambda_user)]
lambda_user <- lambda_user[lambda_user > 0]

if (length(lambda_user) == 0) {
  stop("lambda_user is empty.")
}

##cat("lambda_user length =", length(lambda_user), "\n")
##cat("lambda_user range =", range(lambda_user), "\n")

# Put lambda grid into the author's environment as a fallback.
trans_env$lambda.interval <- lambda_user

# ----------------------------------------------------------------------
# Override PLR_CV().
#
# Why:
# The author's TL() may call PLR_CV() without correctly passing lambda.interval,
# or may pass lambda.interval as the fifth positional argument.
# This wrapper handles both cases.
# ----------------------------------------------------------------------
trans_env$PLR_CV <- local({
  e <- trans_env

  function(
    n,
    x,
    y,
    method,
    tek = 50,
    tol = 0.01,
    kfold = 5,
    lambda.interval = NULL,
    ...
  ) {
    # If a fixed lambda has already been selected from target data,
    # directly use it for source data to save time.
    if (exists("lambda.fixed", envir = e, inherits = FALSE)) {
      lam <- get("lambda.fixed", envir = e)

      fit.final <- e$PLR(
        n = n,
        x = x,
        y = y,
        method = method,
        lambda = lam,
        tek = tek,
        tol = tol
      )

      fit.final$lambda <- lam
      return(fit.final)
    }

    # Otherwise, select lambda by CV. This should only happen for target data.
    if ((is.null(lambda.interval) || length(lambda.interval) == 0) &&
        exists("lambda.interval", envir = e, inherits = FALSE)) {
      lambda.interval <- get("lambda.interval", envir = e)
    }

    lambda.interval <- as.numeric(lambda.interval)
    lambda.interval <- lambda.interval[is.finite(lambda.interval)]
    lambda.interval <- lambda.interval[lambda.interval > 0]

    if (length(lambda.interval) == 0) {
      stop("PLR_CV: lambda.interval is empty.")
    }

    if (kfold > n) {
      kfold <- n
    }

    cat("Fast PLR_CV: lambda grid length =", length(lambda.interval), "\n")

#     set.seed(1)
    fold_id <- sample(rep(seq_len(kfold), length.out = n))

    cv.error <- rep(Inf, length(lambda.interval))

    for (i in seq_along(lambda.interval)) {
      lam <- lambda.interval[i]
      fold.error <- rep(NA, kfold)

      for (jj in seq_len(kfold)) {
        test.idx <- which(fold_id == jj)
        train.idx <- setdiff(seq_len(n), test.idx)

        fit.j <- e$PLR(
          n = length(train.idx),
          x = x[train.idx, , drop = FALSE],
          y = y[train.idx],
          method = method,
          lambda = lam,
          tek = tek,
          tol = tol
        )

        pred <- as.vector(x[test.idx, , drop = FALSE] %*% fit.j$beta)
        fold.error[jj] <- mean((y[test.idx] - pred)^2)
      }

      cv.error[i] <- mean(fold.error, na.rm = TRUE)
    }

    best.lambda <- lambda.interval[which.min(cv.error)]

    cat("Fast PLR_CV selected lambda =", best.lambda, "\n")

    fit.final <- e$PLR(
      n = n,
      x = x,
      y = y,
      method = method,
      lambda = best.lambda,
      tek = tek,
      tol = tol
    )

    fit.final$lambda <- best.lambda
    fit.final$cv.error <- cv.error
    fit.final$lambda.interval <- lambda.interval

    return(fit.final)
  }
})

# ----------------------------------------------------------------------
# Run Trans-PtLR.
#
# We avoid source_detection() here because the author's implementation can
# fail to pass lambda.interval correctly inside nested calls.
# The following is more stable:
#   1. fit target penalized t-regression by CV;
#   2. use TL() to transfer from source data.
# ----------------------------------------------------------------------

##cat("Running target PLR_CV...\n")

model_tar_user <- trans_env$PLR_CV(
  n = targetdata_user$n,
  x = targetdata_user$x,
  y = targetdata_user$y,
  method = method_user,
  tek = 100,
  tol = 0.01,
  kfold = 5,
  lambda.interval = lambda_user
)

##cat("Target PLR_CV finished.\n")
##cat("Selected target lambda =", model_tar_user$lambda, "\n")

##cat("Running TL...\n")

fit <- trans_env$TL(
  targetdata = targetdata_user,
  sourcedatas = sourcedatas_user,
  method = method_user,
  model.tar = model_tar_user,
  lambda.interval = lambda_user
)

##cat("TL finished.\n")

# ----------------------------------------------------------------------
# Extract beta from the output.
# ----------------------------------------------------------------------
extract_beta <- function(obj, p) {
  candidate_names <- c(
    "beta",
    "betahat",
    "beta.hat",
    "beta_hat",
    "coef",
    "coefficients",
    "beta.target",
    "beta_target",
    "beta.tl",
    "beta_TL",
    "TL_beta",
    "betaTL",
    "model",
    "modelTL",
    "model.tar",
    "model_tar"
  )

  if (is.numeric(obj) && length(obj) == p) {
    return(as.numeric(obj))
  }

  if (is.matrix(obj) || is.data.frame(obj)) {
    if (nrow(obj) == p && ncol(obj) == 1) {
      return(as.numeric(obj[, 1]))
    }
    if (ncol(obj) == p && nrow(obj) == 1) {
      return(as.numeric(obj[1, ]))
    }
  }

  if (is.list(obj)) {
    for (nm in candidate_names) {
      if (nm %in% names(obj)) {
        res <- tryCatch(
          extract_beta(obj[[nm]], p),
          error = function(e) NULL
        )
        if (!is.null(res)) {
          return(res)
        }
      }
    }

    for (nm in names(obj)) {
      res <- tryCatch(
        extract_beta(obj[[nm]], p),
        error = function(e) NULL
      )
      if (!is.null(res)) {
        return(res)
      }
    }
  }

  return(NULL)
}

beta_hat <- extract_beta(fit, ncol(X_target))

# If TL() does not return a beta in an easy-to-detect structure,
# try known objects that may be created in the author's environment.
if (is.null(beta_hat)) {
  possible_env_names <- c(
    "betaTL",
    "modelTL",
    "modelTL_naive",
    "beta",
    "model0"
  )

  for (nm in possible_env_names) {
    if (exists(nm, envir = trans_env, inherits = FALSE)) {
      obj <- get(nm, envir = trans_env)
      beta_hat <- tryCatch(
        extract_beta(obj, ncol(X_target)),
        error = function(e) NULL
      )
      if (!is.null(beta_hat)) {
        cat("Extracted beta from trans_env object:", nm, "\n")
        break
      }
    }
  }
}

if (is.null(beta_hat)) {
  cat("fit class:\n")
  print(class(fit))

  cat("fit names:\n")
  print(names(fit))

  cat("fit structure:\n")
  str(fit, max.level = 4)

  cat("Objects inside trans_env:\n")
  print(ls(trans_env))

  stop("Cannot automatically find beta estimate in Trans-PtLR output.")
}

if (length(beta_hat) != ncol(X_target)) {
  stop(paste(
    "Extracted beta has wrong length:",
    length(beta_hat),
    "but expected",
    ncol(X_target)
  ))
}

write.table(
  beta_hat,
  file = file.path(out_dir, "transptlr_beta.csv"),
  row.names = FALSE,
  col.names = FALSE,
  sep = ","
)

cat("Saved beta to:", file.path(out_dir, "transptlr_beta.csv"), "\n")
cat("Trans-PtLR finished successfully.\n")
