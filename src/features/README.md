# src.features

This folder contains the full feature preparation stack used for both **model training** and **inference**.
It is designed to enforce **training–inference parity**, explicit schema alignment, and metadata-driven
feature generation.

The core principle is that *all inference-time transformations must be reproducible from training-time
metadata*, without refitting or implicit assumptions.

---

## High-level Flow

Raw / cleaned data flows through the following stages:

1. **Preprocessing**
   - Type normalization, missing-value handling, and basic cleaning
2. **Feature Engineering**
   - Numeric transforms, categorical encoding, interactions
3. **Feature Expansion (optional)**
   - Geolocation- and amenity-based enrichment
4. **Schema Alignment**
   - Strict column presence and ordering based on training metadata

This flow is shared across batch training, batch evaluation, and single-record inference.

---

## Training vs Inference

Training and inference are explicitly separated but **share metadata contracts**:

- `prepare_features_train_val`
  - Fits encoders, log transforms, OHE columns
  - Produces `meta` describing the full feature schema
- `prepare_features_test` / `transform_single_for_inference`
  - Reuses training metadata only
  - Never refits encoders or derives new feature definitions

All inference paths enforce:
- identical feature names
- identical column order
- safe defaults for missing or unseen categories

---

## Feature Expansion & Enrichment

Feature expansion is handled centrally via `feature_expansion` and supports:

- Row-level ratios and interactions
- Neighborhood- and building-derived features
- Optional geolocation enrichment
- Optional amenities-based proximity features

Geolocation and amenity logic is **fully metadata-driven** and cache-aware, ensuring deterministic
behavior during inference.

---

## Metadata & Schema Guarantees

Training produces a compact metadata object containing:
- numeric and binary feature lists
- log-transformed columns
- categorical OHE columns
- dropped / filtered features
- geolocation and amenity metadata

Inference uses this metadata to:
- add missing columns
- drop unseen features
- deduplicate unsafe inputs
- enforce strict schema consistency

---

## Adding New Features

When introducing new features:

- **Pure row-wise logic** → add in `feature_expansion`
- **Derived numeric / interaction features** → add in feature engineering modules
- **Categorical features** → ensure they are represented in training metadata
- **Inference safety** → confirm defaults are defined when features are missing

All new features must be representable via training metadata.

---

## Design Principles

- Training–inference parity over convenience
- Explicit metadata over implicit state
- Deterministic inference paths
- Defensive handling of real-world input irregularities
