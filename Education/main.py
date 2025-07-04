
from src.data.make_dataset import load_data, preprocess_data
from src.features.build_features import split_features_labels, train_test_split_data
from src.models.train_model import train_xgboost_model, evaluate_model
from src.visualization.visualize import plot_feature_importance
from src.llm.llm_consultant import consult_llm_with_metrics
import joblib

# Load and preprocess the dataset
df = load_data("data/raw/school_data.csv")
df = preprocess_data(df)

# Split into features and labels
X, y = split_features_labels(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Train the model
model = train_xgboost_model(X_train, y_train)
joblib.dump(model, "src/models/xgb_model.pkl")

# Evaluate and collect metrics
metrics = evaluate_model(model, X_test, y_test)

# Use LLM to get insights instead of printing metrics
llm_recommendations = consult_llm_with_metrics(metrics, X.columns.tolist())
print("\n--- LLM-Generated Recommendations ---\n")
print(llm_recommendations)

# Plot feature importance
plot_feature_importance(model)
