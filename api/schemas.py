from pydantic import BaseModel


class PredictRequest(BaseModel):
    home_team:  str
    away_team:  str
    date:       str   # YYYY-MM-DD
    referee:    str
    match_week: int = 20


class ResultPrediction(BaseModel):
    prediction: str
    home_win:   float
    draw:       float
    away_win:   float


class BinaryPrediction(BaseModel):
    prediction:  str
    probability: float


class GoalsPrediction(BaseModel):
    predicted: float


class Predictions(BaseModel):
    result:     ResultPrediction
    btts:       BinaryPrediction
    over_2_5:   BinaryPrediction
    over_1_5:   BinaryPrediction
    total_goals: GoalsPrediction


class PredictResponse(BaseModel):
    match:       str
    date:        str
    predictions: Predictions
