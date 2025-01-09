
    #### **Model Configuration**

    | Metric              | Value                                  |
    |---------------------|----------------------------------------|
    | Best Parameters      | `{k: v for k, v in best_params.items()}` |
    | Best Estimator       | `LinearSVC(C=0.01)`                    |
    | Cross-Val Accuracy   | `0.6`                |
    
    #### **Classification Report**

    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0 | 0.2 | 1.0 | 0.3333333333333333 | 1.0 |
| 1 | 0.0 | 0.0 | 0.0 | 4.0 |
| **Accuracy** |        |        | **0.2**  | 5.0 |
| macro avg | 0.1 | 0.5 | 0.16666666666666666 | 5.0 |
| weighted avg | 0.04 | 0.2 | 0.06666666666666667 | 5.0 |
| **Macro Avg** | 0.1 | 0.5 | 0.16666666666666666 | 5.0 |
| **Weighted Avg** | 0.04 | 0.2 | 0.06666666666666667 | 5.0 |
