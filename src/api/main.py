from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.endpoints import scrape, preprocess, predict, full_pipeline
from src.api.core.manager import PipelineManager
from pathlib import Path
import pandas as pd
import yaml

# ----------------------------------------
# Shared global instance
# ----------------------------------------
manager = PipelineManager()


# ----------------------------------------
# Startup Event: initialize the manager
# ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize the pipeline manager
    manager.initialize(config_dir="config/")
    # Load and inject amenities and geo metadata
    ROOT = Path(__file__).resolve().parents[2]  
    with open(ROOT / "config/preprocessing_config.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    geo_cfg = preprocessing_cfg.get("geo_feature_exp", {})
    manager.pipeline.meta["amenities_df"] = pd.read_csv(
        ROOT / geo_cfg.get("amenities_file")
    )
    manager.pipeline.meta["amenity_radius_map"] = geo_cfg.get(
        "amenity_radius_map"
    )
    manager.pipeline.meta["geo_cache_file"] = str(
        ROOT / geo_cfg.get("geo_cache_file")
    )
    print("[Startup] PipelineManager and amenities loaded successfully.")
    yield
    # Shutdown: optional cleanup
    print("[Shutdown] API shutting down.")


app = FastAPI(
    title="Real Estate Prediction API",
    description="API for scraping, preprocessing, and price prediction.",
    version="0.1.0",
    lifespan=lifespan,
)

# ----------------------------------------
# Register endpoints
# ----------------------------------------
app.include_router(scrape.router, prefix="/scrape", tags=["Scraping"])
app.include_router(
    preprocess.router, prefix="/preprocess", tags=["Preprocessing"]
)
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(
    full_pipeline.router, prefix="/pipeline", tags=["Full Pipeline"]
)


# ----------------------------------------
# Root
# ----------------------------------------
@app.get("/")
def root():
    return {"message": "API is live"}


# ----------------------------------------
# Optional: run with Uvicorn
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)

print(f"Lifespan handler attached? {app.router.lifespan_context is not None}")
