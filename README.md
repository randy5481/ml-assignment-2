## a. Problem Statement

Heart disease is one of the leading causes of death globally. Early and accurate detection can save lives. This project builds and compares multiple machine learning classification models to predict whether a patient is likely to have heart disease, based on clinical and demographic features.

The goal is to evaluate and compare the performance of six different classification algorithms on the same dataset using standard evaluation metrics, and deploy the best-performing pipeline as an interactive web application.

---

## b. Dataset Description

**Dataset:** Heart Failure Prediction Dataset  
**Source:** [Kaggle – Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**License:** Open Database License (ODbL)

| Property | Value |
|----------|-------|
| Total Instances | 918 |
| Total Features | 11 input + 1 target |
| Target Variable | `HeartDisease` (0 = No Disease, 1 = Disease) |
| Task Type | Binary Classification |
| Missing Values | None |

### c. Comparison Table – Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8696 | 0.8971 | 0.8482 | 0.9314 | 0.8879 | 0.7374 |
| Decision Tree | 0.7880 | 0.7861 | 0.8119 | 0.8039 | 0.8079 | 0.5716 |
| K-Nearest Neighbor | 0.8913 | 0.9192 | 0.8942 | 0.9118 | 0.9029 | 0.7797 |
| Naive Bayes | 0.8913 | 0.9280 | 0.8942 | 0.9118 | 0.9029 | 0.7797 |
| Random Forest (Ensemble) | 0.8750 | 0.9229 | 0.8762 | 0.9020 | 0.8889 | 0.7465 |
| XGBoost (Ensemble) | 0.8696 | 0.9230 | 0.8980 | 0.8627 | 0.8800 | 0.7380 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Achieved a solid accuracy of 86.96% with an AUC of 0.8971. It showed a notably high Recall of 0.9314, meaning it correctly identified 93% of actual heart disease cases — which is very valuable in a medical context where missing a positive case (false negative) is costly. The model benefits from well-scaled features and near-linear relationships in the data. However, its relatively lower Precision (0.8482) indicates it produces some false positives. A reliable and interpretable baseline. |
| Decision Tree | Performed the weakest among all models with an accuracy of 78.80% and the lowest AUC of 0.7861. The confusion matrix shows 20 false negatives and 19 false positives, indicating the model struggles with both directions of error. Without pruning or depth constraints, the decision tree tends to overfit the training data and fails to generalize well. The MCC of 0.5716 — significantly lower than all other models — confirms weaker overall discriminative ability. |
| K-Nearest Neighbor | Delivered the best overall performance with 89.13% accuracy, F1 of 0.9029, and MCC of 0.7797. With only 9 false negatives and 11 false positives on 184 test samples, it demonstrates strong and balanced classification. KNN benefits greatly from StandardScaler applied to features since it is distance-based. Its AUC of 0.9192 further confirms excellent class separability. Best suited when training size is manageable and feature scaling is properly applied. |
| Naive Bayes | Tied with KNN on Accuracy (89.13%), Precision, Recall, F1, and MCC, but achieved the highest AUC of 0.9280 among all models — indicating excellent probabilistic calibration. Despite assuming feature independence (which is often violated in medical datasets), Gaussian Naive Bayes performed surprisingly well, suggesting the features carry individually strong predictive signals. Its high AUC makes it particularly suitable for ranking and probability estimation tasks. |
| Random Forest (Ensemble) | Achieved 87.50% accuracy and a strong AUC of 0.9229. As a bagging ensemble of 100 decision trees, it significantly outperformed the single Decision Tree by reducing variance through averaging. The F1 of 0.8889 and MCC of 0.7465 confirm robust and balanced predictions. While it did not top the leaderboard on any single metric, its consistency and interpretability via feature importances make it a practical choice for real-world deployment. |
| XGBoost (Ensemble) | Achieved 86.96% accuracy and a competitive AUC of 0.9230, very close to Random Forest. Notably, XGBoost had the highest Precision (0.8980) among all models — meaning when it predicts heart disease, it is most often correct. However, its Recall of 0.8627 is the lowest — it misses more actual positive cases (14 false negatives vs KNN's 9). In medical applications this trade-off matters; KNN and Logistic Regression are preferable when minimising false negatives is the priority. |
