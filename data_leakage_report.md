# Data Leakage Report

## 1. What Target Leakage Means
Target leakage occurs when a model is trained using features that contain information about the target variable, but these features would not be available in a real-world prediction scenario. If such features are used, the model will appear highly accurate during training and testing, but will fail in production because it relies on information it won't actually have.

## 2. Why `phase.1` is Excluded
The `phase.1` column is a direct duplicate of the target column (`phase`). Including it as a feature would allow the model to simply copy the value of `phase.1` to predict `phase`, perfectly cheating the task. This is the most direct form of target leakage.

## 3. Why `advice_text` is Excluded
The `advice_text` column (and related columns like `advice_type`, `advice_source`, `advice_confidence`) contains post-outcome descriptions or advice that is often manually generated based on the actual phase. These text descriptions indirectly (or sometimes directly) reveal the target phase, which would give the model future information unavailable at the time of prediction.

## 4. Why `id` is Excluded from Model Features
The `id` column uniquely identifies the user. If the model is given the user ID as a feature, it could memorize patterns specific to each individual user rather than learning generalized physiological patterns. This reduces the model's ability to generalize to new, unseen users. 

## 5. Why Group-Based Splitting is Necessary
The dataset contains multiple records (e.g., days) for the same user (`id`). If we use a standard random split, records from the same user could appear in both the training and testing sets. The model might learn person-specific characteristics from the training set and use them to predict on the test set, creating an artificially high accuracy. Group-based splitting ensures that all records for a given user are placed entirely in either the training set or the test set, giving a true estimate of how the model performs on unseen users.

## 6. Why Preprocessing is Fitted Only on Training Data
If we compute imputation values (like median or mode) or feature scalings using the entire dataset before splitting, information from the test set "leaks" into the training process. By fitting the preprocessing pipeline strictly on the training data, we ensure the test data remains completely unseen and untouched until the final evaluation.

## 7. Why the Final Test Set is Not Used for Tuning
Using the test set to repeatedly evaluate and tweak the model parameters would lead to "overfitting to the test set". The model would be tuned to perform well on that specific test data, losing its ability to act as an unbiased evaluator of generalization performance. Therefore, a separate development/validation set (via cross-validation) is used for tuning and model selection.

## 8. Any Other Suspicious Columns Found
- `validator_flag` and `synthetic_augmented_flag` are metadata about how the record was processed and do not represent physiological input from a user.
- Any manual label proxy or post-processed labels were also excluded to ensure the model strictly relies on the raw user inputs provided in tracking apps.
