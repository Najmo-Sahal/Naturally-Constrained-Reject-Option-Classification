This repository reproduces the B-CDF algorithm to handle uncertainty in neural network predictions. It introduces a "natural constraint" that identifies and rejects predictions in regions of random chance classification.  
# Core Workflow
- Synthetic Data: Generated 12 datasets with varying overlap and density.
- Calibration: Applied temperature scaling to minimize Expected Calibration Error (ECE).
- Threshold Learning: Learns per-class softmax thresholds using a binomial statistical test.
- Rejection: Predictions are rejected if confidence falls below the learned threshold, improving the accuracy of accepted examples.
