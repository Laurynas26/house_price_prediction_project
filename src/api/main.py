from fastapi import FastAPI
from src.api.endpoints import scrape

app = FastAPI(
    title="Real Estate Scraper API",
    description="EC2 scraping service for extracting raw listing data.",
    version="0.2.0",
)

# ----------------------------------------
# Register ONLY the scrape endpoint
# ----------------------------------------
app.include_router(scrape.router, prefix="/scrape", tags=["Scraping"])


# ----------------------------------------
# Root
# ----------------------------------------
@app.get("/")
def root():
    return {"message": "Scraper API is live"}


# ----------------------------------------
# Optional: for local development
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
