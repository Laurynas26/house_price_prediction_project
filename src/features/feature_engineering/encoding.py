from sklearn.preprocessing import OrdinalEncoder


def encode_energy_label(X, column="energy_label", encoder=None, fit=True):
    """
    Encode the energy label column with a fixed order using OrdinalEncoder.

    Args:
        X: DataFrame
        column: str, name of the energy label column
        encoder: optional, a pre-fitted OrdinalEncoder (for test/inference)
        fit: if True, fit the encoder; if False, only transform

    Returns:
        X: DataFrame with encoded column added and original dropped
        encoder: fitted OrdinalEncoder
    """
    X = X.copy()

    X[column] = X[column].replace({0: "G"})

    energy_order = [
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
        "A+",
        "A++",
        "A+++",
        "A++++",
    ]

    if encoder is None:
        encoder = OrdinalEncoder(categories=[energy_order])

    if fit:
        X["energy_label_encoded"] = encoder.fit_transform(X[[column]])
    else:
        X["energy_label_encoded"] = encoder.transform(X[[column]])

    X = X.drop(columns=[column])

    return X, encoder
