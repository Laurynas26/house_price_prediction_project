from sklearn.preprocessing import OrdinalEncoder


def encode_energy_label(
    X, column: str = "energy_label", encoder=None, fit: bool = True
):
    """
    Encode the `energy_label` column into an ordinal numeric scale.

    The encoding uses a fixed, domain-specific ordering from lowest ("G") to
    highest efficiency ("A++++"). Handles missing or invalid entries by mapping
    them to "G".

    Parameters
    ----------
    X : pandas.DataFrame
        Input dataframe containing the `energy_label` column.
    column : str, default="energy_label"
        Name of the column to encode.
    encoder : sklearn.preprocessing.OrdinalEncoder, optional
        Pre-fitted encoder. If None, a new encoder is created and (optionally)
        fitted.
    fit : bool, default=True
        If True, fit the encoder on `X[column]`. If False, only transform using
        the provided encoder.

    Returns
    -------
    X : pandas.DataFrame
        DataFrame with a new column `energy_label_encoded` added and the
        original `energy_label` column dropped.
    encoder : sklearn.preprocessing.OrdinalEncoder
        The fitted OrdinalEncoder.
    """
    X = X.copy()
    # Ensure no missing values or unknown categories
    X[column] = X[column].replace({0: "G", "N/A": "G"}).fillna("G")

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
    Encode the `energy_label` column consistently across train, test,
    and optional validation sets.

    The encoder is fitted only on the training data to avoid data leakage,
    then applied to test and validation sets.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training set with `energy_label` column.
    X_test : pandas.DataFrame
        Test set with `energy_label` column.
    X_val : pandas.DataFrame, optional
        Validation set with `energy_label` column. Default is None.

    Returns
    -------
    X_train_encoded : pandas.DataFrame
        Training set with encoded `energy_label`.
    X_test_encoded : pandas.DataFrame
        Test set with encoded `energy_label`.
    X_val_encoded : pandas.DataFrame or None
        Validation set with encoded `energy_label`, or None if not provided.
    encoder : sklearn.preprocessing.OrdinalEncoder
        The fitted OrdinalEncoder (fitted on train set).
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
    """
    Encode the `energy_label` column for training and validation sets.

    The encoder is fitted only on the training set and then applied to
    the validation set, ensuring no leakage.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training set with `energy_label` column.
    X_val : pandas.DataFrame
        Validation set with `energy_label` column.

    Returns
    -------
    X_train_encoded : pandas.DataFrame
        Training set with encoded `energy_label`.
    X_val_encoded : pandas.DataFrame
        Validation set with encoded `energy_label`.
    encoder : sklearn.preprocessing.OrdinalEncoder
        The fitted OrdinalEncoder (fitted on train set).
    """
    X_train_enc, encoder = encode_energy_label(X_train, fit=True)
    X_val_enc, _ = encode_energy_label(X_val, encoder=encoder, fit=False)
    return X_train_enc, X_val_enc, encoder
