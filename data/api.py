from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Retail AI API", version="1.0.0")
DB = Path("retail_ai.db")
MODEL_VERSION = "churn-v1"

class CustomerInput(BaseModel):
    total_orders: int = Field(ge=0)
    total_spent: float
    total_quantity: int

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/api/v1/churn/score")
def score_churn(customer: CustomerInput):
    try:
        from joblib import load
        model = load("churn_model.joblib")
        features = pd.DataFrame([customer.model_dump()])
        score = float(model.predict_proba(features)[0, 1])

        return {
            "churn_probability": score,
            "model_version": MODEL_VERSION,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat()
        }
    except FileNotFoundError:
        raise HTTPException(503, "Churn model has not been saved yet.")

@app.get("/api/v1/anomalies")
def get_anomalies(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(400, "Limit must be between 1 and 1000.")

    if not DB.exists():
        raise HTTPException(503, "Database not found.")

    with sqlite3.connect(DB) as conn:
        try:
            data = pd.read_sql_query(
                """SELECT * FROM transaction_anomalies
                   WHERE anomaly = 'Unusual' LIMIT ?""",
                conn, params=(limit,)
            )
        except Exception:
            raise HTTPException(500, "Could not read anomaly results.")

    return {
        "model_version": "isolation-forest-v1",
        "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(data),
        "results": data.astype(object).where(data.notna(), None).to_dict("records")
    }