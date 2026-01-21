# API Core

This module contains inference-time orchestration logic for the
house price prediction system.

## Design Philosophy

- Inference depends **only on inference-time artifacts**
  (trained model + feature schema)
- Preprocessing pipeline is treated as a best-effort dependency
- Logging must never break prediction

## Key Components

### PipelineManager
Responsible for:
- Loading preprocessing pipeline (optional at inference)
- Loading trained ML model
- Running:
  - scrape → preprocess → predict
  - manual input → preprocess → predict

scrape → preprocess → features → align → predict
                  ↑
          optional pipeline


### Inference Safety
- Missing features are filled with zeros
- Extra features are ignored
- Non-numeric features are either transformed or dropped
- Model feature schema is the single source of truth

### Why this is not split (yet)
PipelineManager intentionally combines concerns to:
- simplify Lambda deployment
- avoid cross-object orchestration bugs
- keep inference logic discoverable

This may be split into:
- PreprocessingManager
- InferenceManager

once the system grows or multiple models are introduced.
