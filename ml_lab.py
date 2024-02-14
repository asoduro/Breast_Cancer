# import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# read data
df = pd.read_csv('data.csv')
# shuffle the data because data in series
df = df.sample(frac=1)
# drop the unnamed and id columns
df = df.drop(columns=['Unnamed: 32', 'id'])

# Separate labels and features
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Convert the M to 1 and B to 0
label = LabelEncoder()
y = label.fit_transform(y)

# Spilt the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN with hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
knn_model = KNeighborsClassifier()
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from the grid search
best_params = grid_search.best_params_

# Final KNN model with best hyperparameters
final_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
final_model.fit(X_train, y_train)

# Evaluate on the test dataset
y_pred_test = final_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Calculate accuracy, sensitivity, and specificity
accuracy = accuracy_score(y_test, y_pred_test)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # True Positive Rate
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # True Negative Rate

print("K-Nearest Neighbors (Final Model): ")
print("Test Set Accuracy: ", accuracy)
print("Sensitivity (True Positive Rate): ", sensitivity)
print("Specificity (True Negative Rate): ", specificity)
print("Best Hyperparameters: ", best_params)
