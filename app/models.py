# app/models.py
from pydantic import BaseModel

# ---- Ehtimallar üçün alt model ----
class Probabilities(BaseModel):
    Cat: float
    Dog: float

# ---- Əsas proqnoz modeli ----
class PredictionOut(BaseModel):
    label: str
    probabilities: Probabilities
    threshold: float

# ---- Səhv cavab sxemi ----
class ErrorOut(BaseModel):
    detail: str
