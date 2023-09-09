import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from my_modules import data_preprocessing, train_random_forest, evaluate_model, generate_submission

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Loading Data
data, test_data, combined = data_preprocessing.load_data()

# Processing Age feature
data, test_data, combined = data_preprocessing.process_age_title_feature(data, test_data, combined)

# Processing Data
X_train, X_test, y_train, y_test = data_preprocessing.process_data(data)

# Create and train the random forest model
model, parametros = train_random_forest.train_random_forest(X_train, y_train)

# Evaluate the model
evaluate_model.evaluate_model(model, X_train, y_train, X_test, y_test, parametros)

# Test Data
X_test_test = data_preprocessing.process_test_data(test_data)

# Generate submission file
generate_submission.generate_submission(model, X_test_test)