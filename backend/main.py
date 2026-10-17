# backend/main.py
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.cloud import firestore
from google.genai import types
from pydantic import BaseModel, Field

# ---------- Config ----------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]  # fail fast if missing
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
HARD_REQUEST_LIMIT = int(os.environ.get("HARD_REQUEST_LIMIT", "5000"))
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5000").split(",")
]

# !! PLACEHOLDER VALUES - replace with the EU default values / benchmarks
# from the official CBAM implementing regulation before real use. !!
DIRECT_FACTOR = {"Steel": 1.9, "Aluminium": 1.6, "Cement": 0.67, "Fertiliser": 2.0}      # tCO2e / t product
PRECURSOR_FACTOR = {"Steel": 1.1, "Aluminium": 0.0, "Cement": 0.0, "Fertiliser": 0.0}    # tCO2e / t precursor (e.g. DRI)
BENCHMARK = {"Steel": 2.3, "Aluminium": 8.0, "Cement": 0.8, "Fertiliser": 2.4}           # tCO2e / t product
PENALTY_MARGIN = 0.10

db = firestore.Client()
gemini = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="CBAMForge AI Backend")


# ---------- Usage guardrail (atomic) ----------
@firestore.transactional
def _bump(txn, ref):
    snap = ref.get(transaction=txn)
    count = (snap.to_dict() or {}).get("count", 0) if snap.exists else 0
    if count >= HARD_REQUEST_LIMIT:
        return False
    txn.set(ref, {"count": count + 1}, merge=True)
    return True


def _check_and_count() -> bool:
    ref = db.collection("system_guardrails").document("cbam_usage_counter")
    return _bump(db.transaction(), ref)


@app.middleware("http")
async def enforce_hard_request_limit(request: Request, call_next):
    # Only count real API calls (not health checks, docs or CORS preflights)
    if request.method == "OPTIONS" or not request.url.path.startswith("/api/"):
        return await call_next(request)
    if not await run_in_threadpool(_check_and_count):
        return JSONResponse(
            status_code=429,
            content={"detail": f"Hard safety limit ({HARD_REQUEST_LIMIT:,} requests) reached."},
        )
    return await call_next(request)


# Added AFTER the guardrail so it wraps it - error responses still get CORS headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------- Models ----------
class CBAMRequest(BaseModel):
    batch_id: str = Field(min_length=1, max_length=64)
    material_type: str  # Steel | Aluminium | Cement | Fertiliser
    total_weight_tons: float = Field(gt=0)
    furnace_kwh_per_ton: float = Field(gt=0)
    grid_factor_tco2_per_mwh: float = Field(default=0.82, gt=0)
    supplier_certificate_text: str = Field(max_length=5000)


class CertExtraction(BaseModel):
    """What Gemini reads out of the supplier certificate (no maths here)."""
    precursor_share_pct: float | None = Field(None, description="% of input that is a high-emission precursor such as DRI. null if not stated")
    verifier: str | None = None
    certificate_ref: str | None = None
    verification_standard: str | None = None
    verified: bool = Field(description="True only if a third-party verification is explicitly stated")
    risk_flags: list[str] = Field(default_factory=list, description="Missing or inconsistent evidence in the certificate")
    summary: str = Field(description="2-3 sentence plain-English audit summary")


# ---------- Helpers ----------
def build_xml(p: CBAMRequest, m: dict, status: str, cert: CertExtraction) -> str:
    # NOTE: namespace/structure is illustrative - align with the official CBAM XML schema.
    root = ET.Element("CBAMReport", xmlns="urn:example:cbam:draft")
    ET.SubElement(root, "GeneratedAt").text = datetime.now(timezone.utc).isoformat()
    item = ET.SubElement(root, "GoodsItem")
    ET.SubElement(item, "BatchID").text = p.batch_id
    ET.SubElement(item, "ProductType").text = p.material_type
    ET.SubElement(item, "Mass", unit="tonnes").text = f"{p.total_weight_tons:.2f}"
    emb = ET.SubElement(item, "EmbeddedEmissions")
    ET.SubElement(emb, "Direct").text = f"{m['direct']:.2f}"
    ET.SubElement(emb, "Indirect").text = f"{m['indirect']:.2f}"
    ET.SubElement(emb, "Precursor").text = f"{m['precursor']:.2f}"
    ET.SubElement(emb, "Total", unit="tCO2e").text = f"{m['total_tco2e']:.2f}"
    ET.SubElement(item, "SpecificEmissions", unit="tCO2e/t").text = f"{m['specific']:.3f}"
    ET.SubElement(
        item, "Verification",
        status="VERIFIED" if cert.verified else "UNVERIFIED",
        verifier=cert.verifier or "", certRef=cert.certificate_ref or "",
    )
    ET.SubElement(item, "ComplianceStatus").text = status
    ET.SubElement(item, "AuditNotes").text = cert.summary
    for flag in cert.risk_flags:
        ET.SubElement(item, "RiskFlag").text = flag
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=False)


# ---------- Routes ----------
@app.get("/")
def health_check():
    return {"status": "online", "project": "CBAMForge Manufacturing Agent"}


@app.post("/api/cbam/audit")
async def audit_cbam_batch(p: CBAMRequest):
    material = p.material_type if p.material_type in DIRECT_FACTOR else "Steel"

    # 1) Gemini only EXTRACTS facts from the certificate text
    prompt = (
        "You are an EU CBAM compliance auditor. Read the supplier certificate notes below and "
        "extract the requested fields. Do not invent values; use null when not stated.\n\n"
        f"Material: {material}\nCertificate notes:\n\"\"\"\n{p.supplier_certificate_text}\n\"\"\""
    )
    resp = await gemini.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CertExtraction,
            temperature=0.1,
        ),
    )
    cert: CertExtraction = resp.parsed or CertExtraction(
        verified=False, risk_flags=["Certificate could not be parsed"], summary="Certificate could not be parsed."
    )

    # 2) The maths is deterministic Python
    w = p.total_weight_tons
    direct = w * DIRECT_FACTOR[material]
    indirect = (p.furnace_kwh_per_ton * w / 1000) * p.grid_factor_tco2_per_mwh
    share = (cert.precursor_share_pct or 0) / 100
    precursor = w * share * PRECURSOR_FACTOR[material]
    total = direct + indirect + precursor
    specific = total / w

    limit = BENCHMARK[material] * (1 + PENALTY_MARGIN)
    compliant = specific <= limit and cert.verified
    status = "Verified Compliant" if compliant else "Review Required"
    metrics = {
        "direct": round(direct, 2), "indirect": round(indirect, 2),
        "precursor": round(precursor, 2), "total_tco2e": round(total, 2),
        "specific": round(specific, 3), "benchmark_limit": round(limit, 3),
    }

    result = {
        "batch_id": p.batch_id,
        "material": material,
        "weight_tons": w,
        "metrics": metrics,
        "ai_audit_report": cert.summary,
        "risk_flags": cert.risk_flags,
        "report_xml": build_xml(p, metrics, status, cert),
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await run_in_threadpool(db.collection("cbam_audits").add, result)
    return result