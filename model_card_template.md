# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created as part of the Udacity Machine Learning DevOps Engineer Nanodegree program.

- Model Type: Random Forest Classifier
- Model Version: 1.0
- Training Date: January 2026
- Framework: scikit-learn
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42

## Intended Use
This model is intended for education purposes as part of a machine learning pipeline deployment project. It demonstrates the complete workflow of training, testing, and deploying a classification model using FastAPI.

- Intended Users: Students, data science practicioners, and machine learning engineers learning about ML deployment and MLOps practices.
- Out-of-Scope Uses: This model should not be used for making actual employment, lending, or other consequential decisions about individuals. The training data is from 1994 and does not reflect current economic conditions.

## Training Data
The model was trained on the Census Income Dataset (also known as "Adult" dataset) from the UCI Machine Learning Repository. The dataset contains demographic information extracted from the 1994 Census database.

- Dataset Size: 32,561 instances
- Training Split: 80% of the data (26,048 instances)
- Features: 14 to include
     - Continuous: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
     - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Target Variable: Binary classification - salary <= 50k or > 50k

Preprocessing
- Categorical features were one-hot encoded using sklearn's OneHotEncoder with handle_unknown="ignore"
- The target variable (salary) was binarized using sklearn's LabelBinarizer
- No scaling was applied to continuous features
- Train-test split performed with 80-20 ratio and random_state=42

## Evaluation Data
The model was evaluated on a held-out test set comprising 20% of the original dataset (6,513 instances). The same preprocessing pipeline used for training data was applied to the test data to ensure consistency.

## Metrics
The model's performance was evaluated using precision, recall and F1 score:

- Precision: 0.7962
- Recall: 0.5372
- F1 Score: 0.6416

Interpretation
- Precision of 0.7962 means that when the model predicts someone earns >50k, it is correct about 80% of the time.
- Recall of 0.5372 means that the model identifies about 54% of  all individuals who actually earn >50k.
- The model is more conservative in predicting high income, prioritizing precision over recall.

Performance on Data Slices
Model performance was evaluated across different education levels to check for potential bias. The slice analysis shows that the model performs differently across education categories, with generally better performance for individuals with higher education levels. Complete slice performance metrics are available in 'slice_output.txt'.

## Ethical Considerations
Bias and Fairness
The model includes demographic features such as race, sex, and native country in its predictions. This creates risk of perpetuating historical biases present in the 1994 Census data. The model should not be used for any decisions that could affect individuals' opportunities or rights.

Privacy
While the Census data is anonymized, demographic information could potentially be used to identify individuals when combined with other dataset.

Temporal Validity
The training data is from 1994 and is significantly outdated. Economic conditions, Income distributions and demographic patterns have changed substantially since then, so the model's predictions may not reflect current realities.

Disparate Impact
Performance variations across demographic groups (as shown in slice analysis) indicate the model may have disparate impact on different populations.

## Caveats and Recommendations
Limitations
- Outdated Data: Training data from 1994 does not reflect current economic conditions, wage distributions or demographic patterns.
- Limited Hyperparameter Tuning: The model uses basic hyperparameters with minimal optimization.
- No Feature Engineering: Uses raw features without sophisticated feature engineering that could improve performance.
- Class Imbalance: The dataset likely has imbalanced classes (more <= 50k than > 50k), which may affect model performance.

Recommendations
- Do not use in production: This model is for educational purposes only and should not be deployed for real-world decision-making.
- Bias auditing required: Any similar model used in practice should undergo thorough bias and fairness auditing. 
- Regular retraining: If a similar model were used in practice, it would require retraining on current data.
- Alternative models: Consider evaluating other algorithms (Gradient Boosting or Nueral Networks) and ensemble methods.
- Feature selection: Carefully consider whether demographic features should be included to avoid discrimination.
- Monitoring: If deployed, implement continuous monitoring for model drift and fairness metrics.

Future Improvements
- Systematic hyperparameter tuning using grid search or random search
- Feature engineering to create more informative predictors
- Addressing class imbalances through resampling or weighted loss functions
- Evaluation with current census data
- Implementation of fairness constraints during training