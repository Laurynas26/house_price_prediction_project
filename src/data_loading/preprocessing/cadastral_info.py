import pandas as pd


def preprocess_cadastral_parcels(parcels):
    if not isinstance(parcels, list):
        return {"num_parcels": 0, "parcels_concat": ""}

    parcel_names = [
        p.get("parcel", "") for p in parcels if isinstance(p, dict)
    ]
    return {
        "num_parcels": len(parcel_names),
        "parcels_concat": "; ".join(parcel_names),
    }


def preprocess_cadastral_column(df):
    df_cad = (
        df["cadastral_parcels"]
        .apply(preprocess_cadastral_parcels)
        .apply(pd.Series)
    )
    df = pd.concat([df.drop(columns=["cadastral_parcels"]), df_cad], axis=1)
    return df


def preprocess_ownership_situations(ownership_list):
    if not isinstance(ownership_list, list) or not ownership_list:
        return {"ownership_type": ""}
    return {"ownership_type": ownership_list[0]}


def preprocess_ownership_column(df):
    df_own = (
        df["ownership_situations"]
        .apply(preprocess_ownership_situations)
        .apply(pd.Series)
    )
    df = pd.concat([df.drop(columns=["ownership_situations"]), df_own], axis=1)
    return df


def preprocess_charges(charges_list):
    if not isinstance(charges_list, list) or not charges_list:
        return {"charges_summary": ""}
    return {"charges_summary": "; ".join(charges_list)}


def preprocess_charges_column(df):
    df_charges = df["charges"].apply(preprocess_charges).apply(pd.Series)
    df = pd.concat([df.drop(columns=["charges"]), df_charges], axis=1)
    return df
