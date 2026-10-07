import os
import logging
from typing import Any, Dict, List, Optional

# 1. Force NumPy and Qdrant eager initialization before DSPy loads
import numpy as np
import qdrant_client

os.environ["DSP_NOTEBOOK_CACHING"] = "False"

import dspy
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 2. Import agent graph builder
from src.agents.retail_agent import build_retail_graph

# Set up logging
logger = logging.getLogger("agentic_platform")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# Structured Error Response Models
# ------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Any] = Field(default=None, description="Additional contextual details")

class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail

# ------------------------------------------------------------------
# FastAPI Initialization & Exception Handlers
# ------------------------------------------------------------------

app = FastAPI(
    title="Agentic Spatial Analytics API",
    description="Production-grade FastAPI service running LangGraph stateful agents over Qdrant & Polars.",
    version="1.0.0"
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles explicit HTTPExceptions raised within endpoints."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            )
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic payload validation errors (422 Unprocessable Entity)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Invalid request payload format or parameters",
                details=exc.errors()
            )
        ).model_dump()
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unhandled internal server errors (500)."""
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred while processing your request.",
            )
        ).model_dump()
    )

# ------------------------------------------------------------------
# DSPy Configuration & Module Setup
# ------------------------------------------------------------------

lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "your-api-key"))
dspy.configure(lm=lm)

class QueryAnalyzer(dspy.Signature):
    """Extract spatial and retail constraints from raw customer intent."""

    customer_query: str = dspy.InputField(desc="Raw unstructured user search query")
    cuisine_type: str = dspy.OutputField(desc="Primary category or food style specified")
    max_walk_time_min: int = dspy.OutputField(desc="Maximum allowable walk time extracted")
    key_constraints: List[str] = dspy.OutputField(desc="Additional implicit or explicit constraints")

class SpatialQueryAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(QueryAnalyzer)

    def forward(self, query: str):
        return self.analyze(customer_query=query)

query_agent = SpatialQueryAgent()
agent_app = build_retail_graph()

# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Looking for casual Italian dining within a 10 min walk"})

class CandidatePOI(BaseModel):
    poi_id: int
    name: str
    cuisine: str
    walk_time_min: int
    rating: float

class SearchResponse(BaseModel):
    extracted_preferences: Dict[str, Any]
    retrieved_locations: List[CandidatePOI]
    execution_trace: List[str]
    is_sufficient: bool

class AnalyzeQueryResponse(BaseModel):
    cuisine_type: str
    max_walk_time_min: int
    key_constraints: List[str]
    reasoning: str

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/api/v1/recommend", response_model=SearchResponse)
async def get_spatial_recommendation(request: SearchRequest):
    initial_input = {
        "customer_query": request.query,
        "iteration_count": 0,
        "execution_trace": []
    }
    result = await agent_app.ainvoke(initial_input)
    return SearchResponse(
        extracted_preferences=result["extracted_preferences"],
        retrieved_locations=result["retrieved_locations"],
        execution_trace=result["execution_trace"],
        is_sufficient=result["is_sufficient"]
    )

@app.post("/api/v1/analyze-query", response_model=AnalyzeQueryResponse)
async def analyze_query_with_dspy(request: SearchRequest):
    prediction = query_agent(query=request.query)
    return AnalyzeQueryResponse(
        cuisine_type=prediction.cuisine_type,
        max_walk_time_min=int(prediction.max_walk_time_min),
        key_constraints=prediction.key_constraints,
        reasoning=prediction.rationale
    )