# Load necessary libraries
# If not already installed, run:
# install.packages("nnet")
# install.packages("knitr")
# install.packages("optimx") # For optimization in temperature scaling
library(nnet)
library(knitr)
library(optimx) # Used for optimizing temperature scaling parameter

# --- 1. Data Loading Function ---

load_dataset_from_csv <- function(dataset_name, run_idx, base_path = "C:/Users/hp/Documents/S4D_Project/") {
  #' Loads training, validation, and test data for a specific dataset and run from CSV files.
  #'
  #' Args:
  #'   dataset_name (character): Name of the dataset (e.g., "Synthetic_1A").
  #'   run_idx (numeric): The run index (e.g., 0 for "run0").
  #'   base_path (character): Base directory where CSV files are stored.
  #'
  #' Returns:
  #'   list: A list containing 'train', 'val', 'test' data, where each is a list
  #'         with 'X' (features matrix) and 'y' (labels vector).
  
  # Construct file paths
  train_file <- file.path(base_path, sprintf("%s_run%d_train.csv", dataset_name, run_idx))
  val_file <- file.path(base_path, sprintf("%s_run%d_val.csv", dataset_name, run_idx))
  test_file <- file.path(base_path, sprintf("%s_run%d_test.csv", dataset_name, run_idx))
  
  # Read CSV files
  # IMPORTANT: Adjust 'sep' and 'header' parameters based on your actual CSV file format.
  # Based on your screenshot, header=TRUE and sep="," are correct.
  train_data_raw <- read.csv(train_file, header = TRUE, sep = ",")
  val_data_raw <- read.csv(val_file, header = TRUE, sep = ",")
  test_data_raw <- read.csv(test_file, header = TRUE, sep = ",")
  
  # Remove any potential empty rows that might be read from the end of the file
  train_data_raw <- train_data_raw[complete.cases(train_data_raw), ]
  val_data_raw <- val_data_raw[complete.cases(val_data_raw), ]
  test_data_raw <- test_data_raw[complete.cases(test_data_raw), ]
  
  # Separate features (X) and labels (y)
  # Last column is assumed to be the class label
  get_X_y <- function(df) {
    num_cols <- ncol(df)
    # Ensure there's at least one feature column before the label
    if (num_cols < 2) {
      stop("CSV file must contain at least one feature column and one label column.")
    }
    # The last column is the label, so features are all columns except the last one
    X <- as.matrix(df[, 1:(num_cols - 1)])
    y <- as.numeric(df[, num_cols])
    return(list(X = X, y = y))
  }
  
  return(list(
    train = get_X_y(train_data_raw),
    val = get_X_y(val_data_raw),
    test = get_X_y(test_data_raw)
  ))
}

# --- 2. Neural Network Training Function ---

train_neural_network <- function(train_data, val_data, num_classes, run_seed) {
  #' Trains a simple neural network and returns its outputs for calibration.
  #'
  #' Args:
  #'   train_data (list): A list with 'X' (features matrix) and 'y' (labels vector) for training.
  #'   val_data (list): A list with 'X' (features matrix) and 'y' (labels vector) for validation.
  #'   num_classes (numeric): The number of output classes for the network.
  #'   run_seed (numeric): A seed for reproducibility of the network initialization.
  #'
  #' Returns:
  #'   list: Contains the trained model, raw softmax predictions (probabilities),
  #'         argmax predictions, and true labels for the validation set.
  
  set.seed(run_seed)
  
  X_train_matrix <- train_data$X
  y_train_factor <- as.factor(train_data$y)
  
  X_val_matrix <- val_data$X
  y_val <- val_data$y
  
  # Ensure feature matrices have consistent column names for nnet's formula interface
  feature_names <- paste0("V", 1:ncol(X_train_matrix))
  colnames(X_train_matrix) <- feature_names
  colnames(X_val_matrix) <- feature_names
  
  train_df <- as.data.frame(X_train_matrix)
  
  model <- nnet(y_train_factor ~ ., data = train_df,
                size = 10,
                maxit = 50,
                linout = FALSE,
                trace = FALSE)
  
  # Get raw softmax predictions (probabilities) for validation set
  val_softmax_preds_raw <- predict(model, newdata = as.data.frame(X_val_matrix), type = "raw")
  
  # FIX: Ensure val_softmax_preds has num_classes columns.
  # nnet for binary classification (num_classes=2) outputs a single column for P(class=1).
  # We need to convert it to two columns [P(class=0), P(class=1)].
  if (num_classes == 2 && ncol(val_softmax_preds_raw) == 1) {
    val_softmax_preds <- cbind(1 - val_softmax_preds_raw, val_softmax_preds_raw)
    colnames(val_softmax_preds) <- c("0", "1") # Assign column names for clarity
  } else if (ncol(val_softmax_preds_raw) != num_classes) {
    # This scenario should ideally not happen for multi-class > 2, but as a safeguard
    stop(sprintf("Mismatch in softmax prediction columns (%d) and num_classes (%d).",
                 ncol(val_softmax_preds_raw), num_classes))
  } else {
    val_softmax_preds <- val_softmax_preds_raw
  }
  
  # Get argmax predictions (0-indexed)
  val_argmax_preds <- apply(val_softmax_preds, 1, which.max) - 1
  
  # Calculate base validation accuracy (before any rejection)
  base_val_accuracy <- mean(val_argmax_preds == y_val)
  
  return(list(model = model,
              base_val_accuracy = base_val_accuracy,
              val_softmax_preds = val_softmax_preds, # Now guaranteed to have num_classes columns
              val_argmax_preds = val_argmax_preds,
              y_val = y_val)) # Include true labels for convenience
}

# --- 3. Temperature Scaling (Calibration) Functions ---

# Helper to compute Negative Log-Likelihood (Cross-Entropy)
calculate_nll <- function(logits, true_labels_one_hot) {
  # Apply softmax to logits
  exp_logits <- exp(logits)
  softmax_probs <- exp_logits / rowSums(exp_logits)
  
  # Clip probabilities to avoid log(0)
  softmax_probs <- pmax(softmax_probs, .Machine$double.eps)
  
  # Calculate NLL
  # FIX: Ensure element-wise multiplication works by making sure dimensions match.
  # This is handled by ensuring softmax_probs has the correct number of columns.
  nll <- -sum(true_labels_one_hot * log(softmax_probs)) / nrow(true_labels_one_hot)
  return(nll)
}

# Function to optimize Temperature T
optimize_temperature <- function(val_softmax_preds, val_true_labels, num_classes) {
  #' Optimizes the temperature (T) for calibration using validation data.
  #'
  #' Args:
  #'   val_softmax_preds (matrix): Raw softmax probabilities from the model on validation data.
  #'   val_true_labels (numeric): True labels for validation data (0-indexed).
  #'   num_classes (numeric): The total number of classes.
  #'
  #' Returns:
  #'   numeric: The optimized temperature T.
  
  # Convert true labels to one-hot encoding
  val_true_labels_one_hot <- matrix(0, nrow = length(val_true_labels), ncol = num_classes)
  for (i in 1:length(val_true_labels)) {
    if (val_true_labels[i] >= 0 && val_true_labels[i] < num_classes) {
      val_true_labels_one_hot[i, val_true_labels[i] + 1] <- 1 # +1 for R's 1-indexing
    } else {
      warning(sprintf("Invalid class label %d found in validation data. Skipping one-hot encoding for this sample.", val_true_labels[i]))
    }
  }
  
  # Convert probabilities to pseudo-logits (inverse of softmax with T=1)
  pseudo_logits <- log(pmax(val_softmax_preds, .Machine$double.eps))
  
  # Objective function for optimization: minimize NLL
  objective_fn <- function(T_param) {
    if (T_param <= 0) { # Temperature must be positive
      return(Inf)
    }
    calibrated_logits <- pseudo_logits / T_param
    return(calculate_nll(calibrated_logits, val_true_labels_one_hot))
  }
  
  # Use optimx for optimization, starting with T=1
  # Method "L-BFGS-B" allows for bounds (T > 0)
  opt_result <- optimx(par = c(T = 1.0), fn = objective_fn, method = "L-BFGS-B", lower = 0.001)
  
  # Extract optimized temperature
  optimized_T <- opt_result$T
  
  return(optimized_T)
}

apply_temperature_scaling <- function(softmax_preds, temperature) {
  #' Applies temperature scaling to raw softmax probabilities.
  #'
  #' Args:
  #'   softmax_preds (matrix): Raw softmax probabilities.
  #'   temperature (numeric): The learned temperature value.
  #'
  #' Returns:
  #'   matrix: Calibrated softmax probabilities.
  
  # Convert probabilities to pseudo-logits
  pseudo_logits <- log(pmax(softmax_preds, .Machine$double.eps))
  
  # Apply temperature scaling to pseudo-logits
  calibrated_logits <- pseudo_logits / temperature
  
  # Apply softmax to get calibrated probabilities
  exp_calibrated_logits <- exp(calibrated_logits)
  calibrated_probs <- exp_calibrated_logits / rowSums(exp_calibrated_logits)
  
  return(calibrated_probs)
}

# --- 4. B-CDF Threshold Learning Function (Modified for calibrated scores) ---

calculate_roc_metrics <- function(predictions, true_labels, max_softmax_scores, per_class_thresholds) {
  #' Calculates Select Accuracy, Reject Accuracy, and Coverage based on per-class thresholds.
  #'
  #' Args:
  #'   predictions (numeric): The argmax class predictions for each example (0-indexed).
  #'   true_labels (numeric): The true class labels for each example (0-indexed).
  #'   max_softmax_scores (numeric): The maximum softmax score for each example.
  #'   per_class_thresholds (numeric): A vector of thresholds, one for each class.
  #'
  #' Returns:
  #'   list: A list containing 'select_accuracy', 'reject_accuracy', and 'coverage'.
  
  is_selected <- rep(TRUE, length(predictions)) # Initialize all as selected
  
  # Determine which examples are rejected based on per-class thresholds
  for (i in 1:length(predictions)) {
    predicted_class <- predictions[i]
    # Ensure predicted_class is within valid bounds for per_class_thresholds
    if (predicted_class + 1 > length(per_class_thresholds) || predicted_class < 0) {
      next # Skip if predicted class is out of bounds for defined thresholds
    }
    threshold_for_class <- per_class_thresholds[predicted_class + 1] # +1 for R's 1-indexing
    
    if (max_softmax_scores[i] <= threshold_for_class) {
      is_selected[i] <- FALSE # Mark as rejected
    }
  }
  
  # Calculate Coverage
  coverage <- mean(is_selected)
  
  # Calculate Select Accuracy
  selected_indices <- which(is_selected)
  if (length(selected_indices) > 0) {
    selected_predictions <- predictions[selected_indices]
    selected_true_labels <- true_labels[selected_indices]
    select_accuracy <- mean(selected_predictions == selected_true_labels)
  } else {
    select_accuracy <- NA # No examples selected
  }
  
  # Calculate Reject Accuracy
  rejected_indices <- which(!is_selected)
  if (length(rejected_indices) > 0) {
    rejected_predictions <- predictions[rejected_indices]
    rejected_true_labels <- true_labels[rejected_indices]
    reject_accuracy <- mean(rejected_predictions == rejected_true_labels)
  } else {
    reject_accuracy <- NA # No examples rejected
  }
  
  return(list(select_accuracy = select_accuracy, reject_accuracy = reject_accuracy, coverage = coverage))
}

binomial_cdf_check <- function(k_star, n, delta) {
  #' Checks the Binomial randomness constraint.
  #' A reject region is feasible if P(K <= k_star) <= 1 - delta,
  #' where K is the number of successes in n trials with p=0.5.
  #'
  #' Args:
  #'   k_star (numeric): Number of observed correct classifications in the reject region.
  #'   n (numeric): Total number of examples in the reject region.
  #'   delta (numeric): Significance level (e.g., 0.05).
  #'
  #' Returns:
  #'   logical: TRUE if the reject region is feasible (random enough), FALSE otherwise.
  
  if (n == 0) {
    return(TRUE) # An empty reject region is considered feasible
  }
  # pbinom calculates P(X <= q)
  p_value <- pbinom(q = k_star, size = n, prob = 0.5)
  return(p_value <= (1 - delta))
}

learn_bcdf_thresholds <- function(calibrated_softmax_preds, val_argmax_preds, y_val, num_classes, delta) {
  #' Learns per-class softmax rejection thresholds using the B-CDF approach.
  #' Now uses calibrated softmax predictions.
  #'
  #' Args:
  #'   calibrated_softmax_preds (matrix): Calibrated softmax probabilities.
  #'   val_argmax_preds (numeric): Argmax predictions (0-indexed).
  #'   y_val (numeric): True labels (0-indexed).
  #'   num_classes (numeric): The number of output classes.
  #'   delta (numeric): The significance level for the Binomial constraint.
  #'
  #' Returns:
  #'   list: A list containing the learned per-class thresholds and the final
  #'         select accuracy, reject accuracy, and coverage for these thresholds.
  
  max_sm_vals <- apply(calibrated_softmax_preds, 1, max)
  
  thresholds <- rep(0, num_classes)
  
  for (c in 0:(num_classes - 1)) {
    class_pred_indices <- which(val_argmax_preds == c)
    
    if (length(class_pred_indices) == 0) {
      next 
    }
    
    c_max_sm <- max_sm_vals[class_pred_indices]
    c_true_labels <- y_val[class_pred_indices]
    c_argmax_preds <- val_argmax_preds[class_pred_indices]
    
    incorrect_idx_in_class_pred <- which(c_argmax_preds != c_true_labels)
    candidate_thresholds <- unique(c_max_sm[incorrect_idx_in_class_pred])
    candidate_thresholds <- sort(candidate_thresholds)
    
    best_thresh <- 0
    initial_selected_in_class_correct <- sum(c_argmax_preds == c_true_labels)
    initial_selected_in_class_total <- length(c_argmax_preds)
    if (initial_selected_in_class_total > 0) {
      best_sa <- initial_selected_in_class_correct / initial_selected_in_class_total
    } else {
      best_sa <- 0
    }
    
    for (thresh in candidate_thresholds) {
      selected_in_class_indices <- which(c_max_sm > thresh)
      
      if (length(selected_in_class_indices) == 0) {
        next
      }
      
      global_reject_indices_for_this_thresh <- which(val_argmax_preds == c & max_sm_vals <= thresh)
      reject_total <- length(global_reject_indices_for_this_thresh)
      reject_corrects <- sum(y_val[global_reject_indices_for_this_thresh] == val_argmax_preds[global_reject_indices_for_this_thresh])
      
      if (binomial_cdf_check(reject_corrects, reject_total, delta)) {
        current_selected_in_class_correct <- sum(c_true_labels[selected_in_class_indices] == c_argmax_preds[selected_in_class_indices])
        current_selected_in_class_total <- length(selected_in_class_indices)
        current_sa <- current_selected_in_class_correct / current_selected_in_class_total
        
        if (!is.na(current_sa) && (current_sa > best_sa || (current_sa == best_sa && thresh < best_thresh))) {
          best_sa <- current_sa
          best_thresh <- thresh
        }
      }
    }
    thresholds[c + 1] <- best_thresh
  }
  
  final_metrics <- calculate_roc_metrics(val_argmax_preds, y_val, max_sm_vals, thresholds)
  
  return(list(per_class_thresholds = thresholds,
              select_accuracy = final_metrics$select_accuracy,
              reject_accuracy = final_metrics$reject_accuracy,
              coverage = final_metrics$coverage))
}

# --- 5. Main Evaluation Loop ---

run_full_evaluation <- function(base_path = "C:/Users/hp/Documents/S4D_Project/", num_runs = 10, delta_values = c(0.05, 0.1, 0.5, 0.75, 0.95)) {
  #' Orchestrates the full evaluation pipeline for all synthetic datasets.
  #'
  #' Args:
  #'   base_path (character): Base directory for loading data.
  #'   num_runs (numeric): Number of runs (random initializations) to average over.
  #'   delta_values (numeric): Vector of delta values for B-CDF.
  #'
  #' Returns:
  #'   list: Nested list of all evaluation results.
  
  all_evaluation_results <- list()
  
  # Define configurations for the 4 base datasets (varying overlap and number of classes)
  # These configurations are used to explicitly pass num_classes to functions.
  dataset_configs <- list(
    Synthetic_1A = list(num_classes = 2),
    Synthetic_1B = list(num_classes = 2),
    Synthetic_1C = list(num_classes = 2),
    Synthetic_2A = list(num_classes = 2),
    Synthetic_2B = list(num_classes = 2),
    Synthetic_2C = list(num_classes = 2),
    Synthetic_3A = list(num_classes = 3),
    Synthetic_3B = list(num_classes = 3),
    Synthetic_3C = list(num_classes = 3),
    Synthetic_4A = list(num_classes = 4),
    Synthetic_4B = list(num_classes = 4),
    Synthetic_4C = list(num_classes = 4)
  )
  
  for (run_idx in 0:(num_runs - 1)) {
    cat(sprintf("\n--- Starting Full Evaluation for Run %d/%d ---\n", run_idx + 1, num_runs))
    run_results <- list()
    
    for (ds_name in names(dataset_configs)) { # Iterate through defined dataset names
      cat(sprintf("  Processing Dataset: %s (Run %d)\n", ds_name, run_idx))
      
      # Load data for the current dataset and run
      data_splits <- load_dataset_from_csv(ds_name, run_idx, base_path)
      train_data <- data_splits$train
      val_data <- data_splits$val
      
      # FIX: Use num_classes from dataset_configs for consistency
      num_classes <- dataset_configs[[ds_name]]$num_classes
      
      # Train Neural Network
      nn_output <- train_neural_network(train_data, val_data, num_classes, run_idx)
      
      # Temperature Scaling (Calibration)
      cat("    Calibrating model with Temperature Scaling...\n")
      # FIX: Pass num_classes to optimize_temperature
      optimized_T <- optimize_temperature(nn_output$val_softmax_preds, nn_output$y_val, num_classes)
      calibrated_val_softmax_preds <- apply_temperature_scaling(nn_output$val_softmax_preds, optimized_T)
      
      # Get argmax predictions from calibrated probabilities (for B-CDF)
      calibrated_val_argmax_preds <- apply(calibrated_val_softmax_preds, 1, which.max) - 1
      
      # Store base model accuracy for comparison (before rejection)
      ds_run_results <- list(
        base_val_accuracy = nn_output$base_val_accuracy,
        temperature = optimized_T,
        bcdf_results = list()
      )
      
      # Run B-CDF for each delta value
      for (delta_val in delta_values) {
        cat(sprintf("      Running B-CDF for delta = %.2f\n", delta_val))
        bcdf_output <- learn_bcdf_thresholds(
          calibrated_val_softmax_preds, # Use calibrated scores
          calibrated_val_argmax_preds,  # Use argmax from calibrated scores
          nn_output$y_val,              # True labels
          num_classes,                  # Pass num_classes
          delta_val
        )
        ds_run_results$bcdf_results[[paste0("delta_", delta_val)]] <- bcdf_output
      }
      run_results[[ds_name]] <- ds_run_results
    }
    all_evaluation_results[[paste0("run_", run_idx)]] <- run_results
  }
  cat("\nFull evaluation pipeline complete.\n")
  return(all_evaluation_results)
}

# --- 6. Results Table Generation and Printing ---

generate_and_print_tables <- function(all_evaluation_results, delta_values = c(0.05, 0.1, 0.5, 0.75, 0.95)) {
  #' Aggregates and prints evaluation results in paper-style tables.
  #'
  #' Args:
  #'   all_evaluation_results (list): The output from run_full_evaluation.
  #'   delta_values (numeric): The delta values used in the evaluation.
  
  cat("\n--- Aggregated Evaluation Results ---\n")
  
  # Initialize storage for all metrics across runs
  metrics_agg <- list()
  
  # Aggregate results from all runs
  for (run_name in names(all_evaluation_results)) {
    for (ds_name in names(all_evaluation_results[[run_name]])) {
      ds_run_data <- all_evaluation_results[[run_name]][[ds_name]]
      
      if (is.null(metrics_agg[[ds_name]])) {
        metrics_agg[[ds_name]] <- list(
          Base = list(SA = c(), RA = c(), Coverage = c())
        )
        for (d in delta_values) {
          metrics_agg[[ds_name]][[paste0("B-CDF_", d * 100)]] <- list(SA = c(), RA = c(), Coverage = c())
        }
      }
      
      # Add Base Model results
      metrics_agg[[ds_name]]$Base$SA <- c(metrics_agg[[ds_name]]$Base$SA, ds_run_data$base_val_accuracy)
      metrics_agg[[ds_name]]$Base$RA <- c(metrics_agg[[ds_name]]$Base$RA, NA) # No rejection for base
      metrics_agg[[ds_name]]$Base$Coverage <- c(metrics_agg[[ds_name]]$Base$Coverage, 1.0) # 100% coverage for base
      
      # Add B-CDF results
      for (d in delta_values) {
        delta_key_bcdf <- paste0("delta_", d)
        metric_key_bcdf <- paste0("B-CDF_", d * 100)
        
        metrics_agg[[ds_name]][[metric_key_bcdf]]$SA <- c(metrics_agg[[ds_name]][[metric_key_bcdf]]$SA, ds_run_data$bcdf_results[[delta_key_bcdf]]$select_accuracy)
        metrics_agg[[ds_name]][[metric_key_bcdf]]$RA <- c(metrics_agg[[ds_name]][[metric_key_bcdf]]$RA, ds_run_data$bcdf_results[[delta_key_bcdf]]$reject_accuracy)
        metrics_agg[[ds_name]][[metric_key_bcdf]]$Coverage <- c(metrics_agg[[ds_name]][[metric_key_bcdf]]$Coverage, ds_run_data$bcdf_results[[delta_key_bcdf]]$coverage)
      }
    }
  }
  
  # Print tables for each dataset
  for (ds_name in names(metrics_agg)) {
    cat(sprintf("\n### Results for Dataset: %s ###\n", ds_name))
    
    table_rows <- list()
    
    # Process Base Model
    base_sa_mean <- mean(metrics_agg[[ds_name]]$Base$SA, na.rm = TRUE) * 100
    base_sa_sd <- sd(metrics_agg[[ds_name]]$Base$SA, na.rm = TRUE) * 100
    base_cov_mean <- mean(metrics_agg[[ds_name]]$Base$Coverage, na.rm = TRUE) * 100
    base_cov_sd <- sd(metrics_agg[[ds_name]]$Base$Coverage, na.rm = TRUE) * 100
    
    table_rows[[length(table_rows) + 1]] <- data.frame(
      Method = "Base",
      SA = sprintf("%.1f (%.1f)", base_sa_mean, base_sa_sd),
      RA = "-", # Paper uses '-' for no rejection
      Coverage = sprintf("%.1f (%.1f)", base_cov_mean, base_cov_sd),
      stringsAsFactors = FALSE
    )
    
    # Process B-CDF results
    for (d in delta_values) {
      metric_key_bcdf <- paste0("B-CDF_", d * 100)
      
      sa_mean <- mean(metrics_agg[[ds_name]][[metric_key_bcdf]]$SA, na.rm = TRUE) * 100
      sa_sd <- sd(metrics_agg[[ds_name]][[metric_key_bcdf]]$SA, na.rm = TRUE) * 100
      ra_mean <- mean(metrics_agg[[ds_name]][[metric_key_bcdf]]$RA, na.rm = TRUE) * 100
      ra_sd <- sd(metrics_agg[[ds_name]][[metric_key_bcdf]]$RA, na.rm = TRUE) * 100
      cov_mean <- mean(metrics_agg[[ds_name]][[metric_key_bcdf]]$Coverage, na.rm = TRUE) * 100
      cov_sd <- sd(metrics_agg[[ds_name]][[metric_key_bcdf]]$Coverage, na.rm = TRUE) * 100
      
      # Handle NaN/NA for RA gracefully
      ra_str <- if (is.nan(ra_mean) || is.na(ra_mean)) "-" else sprintf("%.1f (%.1f)", ra_mean, ra_sd)
      
      table_rows[[length(table_rows) + 1]] <- data.frame(
        Method = sprintf("B-CDF_%.0f", d * 100),
        SA = sprintf("%.1f (%.1f)", sa_mean, sa_sd),
        RA = ra_str,
        Coverage = sprintf("%.1f (%.1f)", cov_mean, cov_sd),
        stringsAsFactors = FALSE
      )
    }
    
    # Combine rows into a data frame and print
    final_table_df <- do.call(rbind, table_rows)
    print(kable(final_table_df, format = "markdown", align = c('l', 'r', 'r', 'r')))
    cat("\n")
  }
}

# --- Example Usage (Uncomment to run) ---
# IMPORTANT: Ensure your CSV files are in the specified base_path.
# Set the base path to your data directory
base_data_path <- "C:/Users/hp/Documents/S4D_Project/"

# Number of runs for averaging (as per paper)
# If you only have run0 files, set num_experiment_runs to 1.
# If you generate all 10 runs, set it to 10.
num_experiment_runs <- 1 # Changed to 1 for immediate testing with only run0 files

# Delta values to test for B-CDF
deltas_to_test <- c(0.05, 0.10, 0.50, 0.75, 0.95)

# Run the full evaluation pipeline
all_results <- run_full_evaluation(
  base_path = base_data_path,
  num_runs = num_experiment_runs,
  delta_values = deltas_to_test
)

# Generate and print the aggregated tables
generate_and_print_tables(all_results, deltas_to_test)
