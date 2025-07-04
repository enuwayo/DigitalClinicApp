# from lightgbm import LGBMRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib
# import numpy as np
# import os

# def train_lgb_model(X, y, model_path):
#     # Split dataset
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train model
#     model = LGBMRegressor(random_state=42)
#     model.fit(X_train, y_train, categorical_feature='auto')

#     # Evaluate model
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ Fixed for older sklearn versions
#     r2 = r2_score(y_test, y_pred)

#     # Save model
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     joblib.dump(model, model_path)

#     return rmse, r2

# def load_trained_model(model_path):
#     """Load the saved LightGBM model."""
#     return joblib.load(model_path)

# def make_predictions(model, X_new):
#     """Make predictions on new data using the loaded model."""
#     return model.predict(X_new)
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import os
import json

def train_lgb_model(X, y, model_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train, categorical_feature='auto')

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Save the feature names used during training
    feature_path = model_path.replace(".pkl", "_features.json")
    with open(feature_path, "w") as f:
        json.dump(X_train.columns.tolist(), f)

    return rmse, r2

def load_trained_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, X_new):
    return model.predict(X_new)
