from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ml_model import predict_startup

app = FastAPI(
    title="Prosperity Prognosticator API",
    version="1.0"
)

# Enable CORS (for frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartupData(BaseModel):
    founded_year: int
    funding_total_usd: float
    age_first_funding_year: Optional[float] = 0
    age_last_funding_year: Optional[float] = 0
    has_VC: bool = False
    has_angel: bool = False
    has_roundA: bool = False
    has_roundB: bool = False
    has_roundC: bool = False
    has_roundD: bool = False

@app.get("/")
def home():
    return {"message": "Startup Success Predictor API Running 🚀"}

@app.post("/predict")
def predict(data: StartupData):
    result = predict_startup(data.dict())
    return result