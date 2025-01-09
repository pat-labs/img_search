
    #### **Model Configuration**

    | Metric              | Value                                  |
    |---------------------|----------------------------------------|
    | Best Parameters      | `{k: v for k, v in best_params.items()}` |
    | Best Estimator       | `LinearSVC(C=0.01)`                    |
    | Cross-Val Accuracy   | `0.7333333333333333`                |
    
    #### **Classification Report**

    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | 0 | 1.0 | 0.3333333333333333 | 0.5 | 3.0 |
| 1 | 0.5 | 1.0 | 0.6666666666666666 | 2.0 |
| **Accuracy** |        |        | **0.6**  | 5.0 |
| macro avg | 0.75 | 0.6666666666666666 | 0.5833333333333333 | 5.0 |
| weighted avg | 0.8 | 0.6 | 0.5666666666666667 | 5.0 |
| **Macro Avg** | 0.75 | 0.6666666666666666 | 0.5833333333333333 | 5.0 |
| **Weighted Avg** | 0.8 | 0.6 | 0.5666666666666667 | 5.0 |
