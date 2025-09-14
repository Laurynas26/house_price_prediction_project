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

    X[column] = X[column].replace({0: "G"}).replace("N/A", "G")

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


def encode_energy_labels_train_test_val(X_train, X_test, X_val=None):
    """
    Encode energy_label feature for train, test, and optional validation sets.

    Returns:
        X_train_encoded, X_test_encoded, X_val_encoded (or None if no val),
        encoder
    """
    X_train_encoded, encoder = encode_energy_label(X_train, fit=True)
    X_test_encoded, _ = encode_energy_label(X_test, encoder=encoder, fit=False)
    if X_val is not None:
        X_val_encoded, _ = encode_energy_label(
            X_val, encoder=encoder, fit=False
        )
    else:
        X_val_encoded = None
    return X_train_encoded, X_test_encoded, X_val_encoded, encoder

def encode_train_val_only(X_train, X_val):
    X_train_enc, encoder = encode_energy_label(X_train, fit=True)
    X_val_enc, _ = encode_energy_label(X_val, encoder=encoder, fit=False)
    return X_train_enc, X_val_enc, encoder

