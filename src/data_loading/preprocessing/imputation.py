def impute_missing_values(df, strategies: dict):
    for col, strategy in strategies.items():
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy.startswith("constant:"):
            value = strategy.split(":", 1)[1]
            df[col] = df[col].fillna(float(value))
        else:
            raise ValueError(f"Unknown imputation strategy: {strategy}")
    return df