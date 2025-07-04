from sklearn.model_selection import train_test_split

def split_features_labels(df, target_col='target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
