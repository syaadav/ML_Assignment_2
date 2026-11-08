### a. Problem Statement
To build a machine learning classification model capable of accurately diagnosing breast cancer based on digitized image features of a fine needle aspirate (FNA) of a breast mass.

### b. Dataset Description
*   **Source:** Scikit-learn built-in Breast Cancer Wisconsin dataset.
*   **Features:** 30 numerical features computing the radius, texture, perimeter, area, smoothness, etc., of cell nuclei.
*   **Instances:** 569 samples.
*   **Target:** Binary classification (0 = Malignant, 1 = Benign).

### c. Github Repository Link
https://github.com/syaadav/ML_Assignment_2

### d. Models Used & Comparison
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9561 | 0.9977 | 0.9569 | 0.9561 | 0.9558 | 0.9068 |
| Decision Tree | 0.9474 | 0.9440 | 0.9474 | 0.9474 | 0.9474 | 0.8880 |
| KNN | 0.9561 | 0.9959 | 0.9590 | 0.9561 | 0.9555 | 0.9086 |
| Naive Bayes | 0.9737 | 0.9984 | 0.9748 | 0.9737 | 0.9735 | 0.9447 |
| Support Vector Machine | 0.9474 | 0.9934 | 0.9515 | 0.9474 | 0.9465 | 0.8907 |

### e. Observations
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Shows strong overall performance with an excellent AUC score and a good balance of precision and recall. |
| Decision Tree | Yields the lowest AUC score among the models, though it still maintains a solid accuracy and balanced F1 score. |
| KNN | Performs excellently with high precision and recall, demonstrating strong capability in identifying the target classes. |
| Naive Bayes | Achieves the highest accuracy, AUC, and MCC scores, making it the most robust and accurate model for this dataset. |
| Support Vector Machine | Delivers good accuracy and a high AUC score, though it is slightly outperformed by Naive Bayes and Logistic Regression on this test set. |
| **Overall, Winner for your dataset?** | **Naive Bayes** |
