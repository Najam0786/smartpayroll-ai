# src/api/main.py
"""
SmartPayroll AI — FastAPI Service
Exposes ML model and investigation agent as REST endpoints.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Request/Response Models ──────────────────────────────────
class AttritionRequest(BaseModel):
    employee_id: int = Field(
        ge=1, le=1470,
        description="Employee ID (1 to 1470)"
    )


class AttritionResponse(BaseModel):
    employee_id: int
    risk_level: str
    risk_score: int
    risk_factors: list[str]
    recommendation: str
    department: str
    monthly_income: float


class InvestigationRequest(BaseModel):
    employee_id: int = Field(
        ge=1, le=1470,
        description="Employee ID to investigate"
    )


class InvestigationResponse(BaseModel):
    employee_id: int
    status: str
    risk_level: str
    summary: str


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


# ─── App Setup ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("SmartPayroll AI API starting...")
    yield
    logger.info("SmartPayroll AI API shutting down.")


app = FastAPI(
    title="SmartPayroll AI API",
    description="AI-powered HR attrition prediction and investigation",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="SmartPayroll AI is running"
    )


@app.post("/api/v1/attrition/risk", response_model=AttritionResponse)
async def get_attrition_risk(request: AttritionRequest):
    """
    Get attrition risk for an employee.

    Returns risk level (LOW/MEDIUM/HIGH),
    risk factors, and recommended action.
    """
    import json
    from src.agents.tools.hr_tools import (
        get_employee_details,
        get_attrition_risk,
    )

    logger.info(f"Attrition risk request for employee {request.employee_id}")

    # Get employee details
    details_raw = get_employee_details(request.employee_id)
    details = json.loads(details_raw)

    if details["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Employee {request.employee_id} not found"
        )

    # Get risk assessment
    risk_raw = get_attrition_risk(request.employee_id)
    risk = json.loads(risk_raw)

    return AttritionResponse(
        employee_id=request.employee_id,
        risk_level=risk["risk_level"],
        risk_score=risk["risk_score"],
        risk_factors=risk["risk_factors"],
        recommendation=risk["recommendation"],
        department=details["department"],
        monthly_income=details["monthly_income"],
    )


@app.post("/api/v1/investigate", response_model=InvestigationResponse)
async def investigate_employee(request: InvestigationRequest):
    """
    Run full investigation for an employee.

    Combines employee details, risk assessment,
    and department comparison into one report.
    """
    from src.agents.investigation_agent import investigate_employee

    logger.info(f"Investigation request for employee {request.employee_id}")

    result = investigate_employee(request.employee_id)

    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Employee {request.employee_id} not found"
        )

    return InvestigationResponse(
        employee_id=request.employee_id,
        status=result["status"],
        risk_level=result["risk"]["risk_level"],
        summary=result["summary"],
    )


@app.get("/api/v1/employees/{employee_id}")
async def get_employee(employee_id: int):
    """Get employee details by ID."""
    import json
    from src.agents.tools.hr_tools import get_employee_details

    if employee_id < 1 or employee_id > 1470:
        raise HTTPException(
            status_code=400,
            detail="Employee ID must be between 1 and 1470"
        )

    result = json.loads(get_employee_details(employee_id))

    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Employee not found")

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )