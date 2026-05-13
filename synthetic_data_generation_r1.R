# --- Helper Functions for Data Generation ---

generate_uniform_2d <- function(num_samples, x_range, y_range) {
  x <- runif(num_samples, min = x_range[1], max = x_range[2])
  y <- runif(num_samples, min = y_range[1], max = y_range[2])
  return(cbind(x, y))
}

generate_gaussian_2d <- function(num_samples, mean, std_dev) {
  x <- rnorm(num_samples, mean = mean[1], sd = std_dev)
  y <- rnorm(num_samples, mean = mean[2], sd = std_dev)
  return(cbind(x, y))
}

.split_data_stratified <- function(X, y, num_classes, num_train_per_class, num_val_per_class, num_test_per_class, density_ratios = NULL) {
  X_train <- list(); y_train <- list()
  X_val <- list(); y_val <- list()
  X_test <- list(); y_test <- list()
  
  for (i in 0:(num_classes - 1)) {
    class_indices <- which(y == i)
    class_indices <- sample(class_indices)
    
    if (!is.null(density_ratios)) {
      scaled_num_train <- as.integer(num_train_per_class * density_ratios[i + 1])
      scaled_num_val <- as.integer(num_val_per_class * density_ratios[i + 1])
      scaled_num_test <- as.integer(num_test_per_class * density_ratios[i + 1])
    } else {
      scaled_num_train <- num_train_per_class
      scaled_num_val <- num_val_per_class
      scaled_num_test <- num_test_per_class
    }
    
    train_idx <- class_indices[1:scaled_num_train]
    val_idx <- class_indices[(scaled_num_train + 1):(scaled_num_train + scaled_num_val)]
    test_idx <- class_indices[(scaled_num_train + scaled_num_val + 1):(scaled_num_train + scaled_num_val + scaled_num_test)]
    
    X_train[[i + 1]] <- X[train_idx, , drop = FALSE]; y_train[[i + 1]] <- y[train_idx]
    X_val[[i + 1]] <- X[val_idx, , drop = FALSE]; y_val[[i + 1]] <- y[val_idx]
    X_test[[i + 1]] <- X[test_idx, , drop = FALSE]; y_test[[i + 1]] <- y[test_idx]
  }
  
  return(list(
    train = list(X = do.call(rbind, X_train), y = unlist(y_train)),
    val = list(X = do.call(rbind, X_val), y = unlist(y_val)),
    test = list(X = do.call(rbind, X_test), y = unlist(y_test))
  ))
}

# --- Saving Function ---
save_dataset_to_csv <- function(data, name_prefix, run_number) {
  dir.create("data", showWarnings = FALSE)
  write.csv(data.frame(data$train$X, y = data$train$y), sprintf("data/%s_run%d_train.csv", name_prefix, run_number), row.names = FALSE)
  write.csv(data.frame(data$val$X, y = data$val$y), sprintf("data/%s_run%d_val.csv", name_prefix, run_number), row.names = FALSE)
  write.csv(data.frame(data$test$X, y = data$test$y), sprintf("data/%s_run%d_test.csv", name_prefix, run_number), row.names = FALSE)
}

# --- Generalized Dataset Creation Function ---

create_synthetic_dataset <- function(dataset_type, num_classes, num_train_per_class, num_val_per_class, num_test_per_class, overlap_strength = 0.5, density_ratios = NULL) {
  all_X <- list(); all_y <- list()
  
  centers <- switch(as.character(num_classes),
                    "2" = list(c(-2, 0), c(2, 0)),
                    "3" = list(c(-3, -2), c(0, 3), c(3, -2)),
                    "4" = list(c(-3, 3), c(3, 3), c(-3, -3), c(3, -3)),
                    stop("num_classes must be 2, 3, or 4.")
  )
  
  if ((dataset_type == 'B' || dataset_type == 'C') && is.null(density_ratios)) {
    density_ratios <- switch(as.character(num_classes),
                             "2" = c(2, 1), "3" = c(4, 2, 1), "4" = c(6, 5, 4, 3))
  } else if (dataset_type == 'A') {
    density_ratios <- rep(1, num_classes)
  } else if ((dataset_type == 'B' || dataset_type == 'C') && length(density_ratios) != num_classes) {
    stop("density_ratios must match num_classes.")
  }
  
  for (i in 0:(num_classes - 1)) {
    center <- centers[[i + 1]]
    total_samples <- as.integer((num_train_per_class + num_val_per_class + num_test_per_class) * density_ratios[i + 1])
    
    if (dataset_type == 'A' || dataset_type == 'B') {
      region_size <- 4.0 * (1 + overlap_strength)
      x_range <- c(center[1] - region_size/2, center[1] + region_size/2)
      y_range <- c(center[2] - region_size/2, center[2] + region_size/2)
      class_X <- generate_uniform_2d(total_samples, x_range, y_range)
    } else if (dataset_type == 'C') {
      class_X <- generate_gaussian_2d(total_samples, center, 1.0 * (1 + overlap_strength))
    }
    
    class_y <- rep(i, total_samples)
    all_X[[i + 1]] <- class_X
    all_y[[i + 1]] <- class_y
  }
  
  X <- do.call(rbind, all_X)
  y <- unlist(all_y)
  return(.split_data_stratified(X, y, num_classes, num_train_per_class, num_val_per_class, num_test_per_class, if (dataset_type != 'A') density_ratios else NULL))
}

# --- Main Data Generation Loop ---

generate_all_synthetic_datasets <- function(num_runs = 10) {
  datasets <- list()
  num_train_per_class <- 1000
  num_val_per_class <- 1000
  num_test_per_class <- 4000
  
  dataset_configs <- list(
    Synthetic_1 = list(num_classes = 2, overlap_strength = 0.1),
    Synthetic_2 = list(num_classes = 2, overlap_strength = 0.7),
    Synthetic_3 = list(num_classes = 3, overlap_strength = 0.3),
    Synthetic_4 = list(num_classes = 4, overlap_strength = 0.6)
  )
  
  density_ratios_map <- list(
    `2` = c(2, 1), `3` = c(4, 2, 1), `4` = c(6, 5, 4, 3)
  )
  
  for (run in 0:(num_runs - 1)) {
    cat(sprintf("Generating data for run %d/%d...\n", run + 1, num_runs))
    run_data <- list()
    set.seed(run)
    
    for (ds_base_name in names(dataset_configs)) {
      config <- dataset_configs[[ds_base_name]]
      num_classes <- config$num_classes
      overlap_strength <- config$overlap_strength
      
      for (variant in c("A", "B", "C")) {
        ds_name <- paste0(ds_base_name, variant)
        density_ratios <- if (variant == "A") NULL else density_ratios_map[[as.character(num_classes)]]
        
        data <- create_synthetic_dataset(variant, num_classes, num_train_per_class, num_val_per_class, num_test_per_class, overlap_strength, density_ratios)
        run_data[[ds_name]] <- data
        
        # 💾 Save to CSV
        save_dataset_to_csv(data, ds_name, run)
      }
    }
    
    datasets[[paste0("run_", run)]] <- run_data
  }
  cat("\n✅ Synthetic data generation complete and saved to ./data/\n")
  return(datasets)
}
