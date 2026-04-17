"""
Soca Scores Prediction API

Endpoints:
    GET  /health          — liveness check
    GET  /teams           — list of all known teams
    POST /predict         — predict all 5 targets for a given match

Run locally:
    uvicorn api.main:app --reload
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import PredictRequest, PredictResponse
from src.components.model_inference import ModelInference
from src.logger import logging


inferencer: ModelInference = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inferencer
    logging.info("Loading models on startup")
    inferencer = ModelInference()
    logging.info("Models ready")
    yield
    logging.info("Shutting down")


app = FastAPI(
    title="Soca Scores — EPL Match Prediction API",
    description="Predicts match result, BTTS, over/under goals, and expected goals for EPL matches.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": inferencer is not None}


@app.get("/teams")
def list_teams():
    teams = sorted(inferencer.team_encoder.classes_.tolist())
    return {"teams": teams, "count": len(teams)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = inferencer.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            date=request.date,
            referee=request.referee,
            match_week=request.match_week,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")
