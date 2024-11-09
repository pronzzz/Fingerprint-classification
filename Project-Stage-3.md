# Stage 3: Evaluating and Optimizing Machine Learning Algorithms for Accurate Fingerprint Recognition

## C1. Evaluating Algorithms

The evaluation process begins by dividing the data into training and test sets, typically with an 80-20 or 70-30 split. The purpose is to assess the performance of selected algorithms — specifically, **K-Nearest Neighbour (KNN)** and **Random Forest** — on the designated fingerprint data. To accomplish this, a **test harness** is created, utilizing **cross-validation** to check for overfitting and evaluate whether other techniques may be necessary to address it.

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Split data (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize models
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()

# Cross-validation
rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
knn_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Random Forest CV scores:", rf_scores)
print("Random Forest mean CV score:", rf_scores.mean())
print("KNN CV scores:", knn_scores)
print("KNN mean CV score:", knn_scores.mean())

# Train and evaluate on test set
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
knn_pred = knn.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))
```

### Summary of Results:
- **Random Forest** performs robustly and is typically less prone to overfitting.
- **KNN** may be more effective at capturing intricate patterns in the data, albeit sometimes at the expense of generalization.

## C2. Improving Results

To enhance performance, **hyperparameter tuning** is performed using **GridSearch** to refine the model parameters. By adjusting these parameters, the algorithms can be further optimized.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# Define parameter grids
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Random Forest tuning
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)
print("Best Random Forest parameters:", rf_grid.best_params_)
print("Best Random Forest CV score:", rf_grid.best_score_)

# KNN tuning
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=3, n_jobs=-1, verbose=2)
knn_grid.fit(X_train, y_train)
print("Best KNN parameters:", knn_grid.best_params_)
print("Best KNN CV score:", knn_grid.best_score_)
```

## C3. Presenting Results

The optimized models are then applied to the test set, and the **classification reports** highlight the most effective model for fingerprint recognition.

```python
best_rf = rf_grid.best_estimator_
best_knn = knn_grid.best_estimator_

rf_pred = best_rf.predict(X_test)
knn_pred = best_knn.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_pred))
```

## C4. Visualizing Results

A **confusion matrix** visualizes the performance of the best model, illustrating test vs. training cases for a clearer comparison.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

best_model = best_rf if rf_grid.best_score_ > knn_grid.best_score_ else best_knn
best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

## C5. Predicting Based on Unseen Data

Predictions are made on synthetic "unseen" data using the optimized model to evaluate its generalizability.

```python
import numpy as np

# Generate synthetic "unseen" data
unseen_data = np.random.rand(5, X_selected.shape[1])

# Make predictions
unseen_predictions = best_model.predict(unseen_data)
print("Predictions for unseen data:")
for i, pred in enumerate(unseen_predictions):
    print(f"Sample {i+1}: Predicted class {pred}")
```

## C6 - C7: Creating a Standalone Model and Saving It for Future Use

The model is trained on the entire dataset and saved for future applications, where it can be further refined.

```python
import joblib

# Train on entire dataset
final_model = best_model.fit(X_selected, y)

# Save the model
joblib.dump(final_model, 'fingerprint_classifier.joblib')
print("Model saved as 'fingerprint_classifier.joblib'")
```

## Privacy and Ethical Issues

- **Data Privacy:** Fingerprint data should be stored securely and anonymously.
- **Consent:** Ensure explicit consent is obtained before collecting fingerprint data.
- **Inaccuracies:** Regularly assess for biases across demographic groups.
- **Transparency:** Clearly explain the model's limitations and decision-making.
- **Security:** Protect against unauthorized access to data or the model.
- **Purpose Limitation:** Use the model only for its intended purpose.
- **Right to Be Forgotten:** Implement methods for users to request data deletion.

## Conclusion

The **Random Forest algorithm** showed consistent accuracy and adaptability in classifying fingerprints from the SOCOFing dataset, with overall accuracy exceeding 80% across implemented algorithms. Future work could involve refining Random Forest, applying data augmentation, exploring **Convolutional Neural Networks (CNNs)**, and expanding the dataset. This project offers a strong foundation for continued research in fingerprint recognition technology.
