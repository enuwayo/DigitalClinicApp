def preprocess_data(df, training=True):
    df = df.copy()

    # Drop 'Date' if present
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    # Ensure categorical columns exist
    categorical_cols = [col for col in ['type', 'region'] if col in df.columns]
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    if training:
        X = df.drop(columns=['AveragePrice'])
        y = df['AveragePrice']
        return X, y
    else:
        return df, None
