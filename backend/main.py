# backend/main.py
import os
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import firestore
from google import genai
from pydantic import BaseModel

app = FastAPI(title="CBAMForge AI Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = firestore.Client()
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

HARD_REQUEST_LIMIT = 5000

@app.middleware("http")
async def enforce_hard_request_limit(request: Request, call_next):
    if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    counter_ref = db.collection("system_guardrails").document("cbam_usage_counter")
    doc = counter_ref.get()
    current_count = doc.to_dict().get("count", 0) if doc.exists else 0

    if current_count >= HARD_REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Hard safety limit (5,000 requests) reached. Serving cached compliance audit."
        )

    counter_ref.set({"count": firestore.Increment(1)}, merge=True)
    return await call_next(request)

@app.get("/")
def health_check():
    return {"status": "online", "project": "CBAMForge Manufacturing Agent"}

class CBAMRequest(BaseModel):
    batch_id: str
    material_type: str  # e.g., "Steel" or "Aluminium"
    total_weight_tons: float
    furnace_kwh_per_ton: float
    supplier_certificate_text: str

@app.post("/api/cbam/audit")
async def audit_cbam_batch(payload: CBAMRequest):
    prompt = f"""
    You are an expert EU CBAM (Carbon Border Adjustment Mechanism) compliance auditor.
    Analyze the following production batch data and supplier certificate:
    - Batch ID: {payload.batch_id}
    - Material: {payload.material_type}
    - Total Weight: {payload.total_weight_tons} metric tons
    - Furnace Energy Consumption: {payload.furnace_kwh_per_ton} kWh/ton
    - Supplier Certificate Notes: {payload.supplier_certificate_text}

    Calculate:
    1. Direct embedded emissions (tCO2e) based on standard grid/furnace emission factors.
    2. Indirect precursor emissions.
    3. Total specific embedded carbon (tCO2e per ton).
    4. Compliance risk status against the 2026 EU CBAM default benchmark + 10% penalty threshold.
    5. Formulate a structured XML snippet or audit summary ready for EU registry submission.
    """

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    audit_result = {
        "batch_id": payload.batch_id,
        "material": payload.material_type,
        "weight_tons": payload.total_weight_tons,
        "ai_audit_report": response.text,
        "status": "Verified Compliant" if "compliant" in response.text.lower() else "Review Required",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    db.collection("cbam_audits").add(audit_result)
    return audit_result