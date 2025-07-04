import matplotlib.pyplot as plt
import xgboost as xgb

def plot_feature_importance(model, max_num_features=10):
    xgb.plot_importance(model, max_num_features=max_num_features)
    plt.show()
