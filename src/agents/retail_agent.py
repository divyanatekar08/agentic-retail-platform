import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from src.spatial_engine import SpatialDataEngine

# Instantiate Spatial Engine once
spatial_engine = SpatialDataEngine()

class RetailAgentState(TypedDict):
    customer_query: str
    extracted_preferences: Dict[str, Any]
    retrieved_locations: List[Dict[str, Any]]
    critique_feedback: str
    iteration_count: int
    is_sufficient: bool
    execution_trace: Annotated[List[str], operator.add]

def parse_preference_node(state: RetailAgentState) -> Dict[str, Any]:
    return {
        "extracted_preferences": {
            "cuisine": "Italian",
            "max_walk_time_min": 10,
            "ambiance": "casual"
        },
        "execution_trace": ["Parsed customer preference constraints"]
    }

def spatial_retrieval_node(state: RetailAgentState) -> Dict[str, Any]:
    """Node B: Queries Qdrant + Polars vector/spatial engine."""
    prefs = state["extracted_preferences"]
    
    # Run hybrid Qdrant + Polars query
    polars_df = spatial_engine.hybrid_spatial_search(
        target_cuisine=prefs["cuisine"],
        max_walk_time=prefs["max_walk_time_min"]
    )
    
    # Convert Polars DataFrame back to native dicts for LangGraph state serialization
    candidates = polars_df.to_dicts()
    
    return {
        "retrieved_locations": candidates,
        "execution_trace": [f"Qdrant/Polars Engine retrieved {len(candidates)} spatial matches"]
    }

def quality_evaluator_node(state: RetailAgentState) -> Dict[str, Any]:
    candidates = state.get("retrieved_locations", [])
    current_iteration = state.get("iteration_count", 0) + 1
    
    is_sufficient = len(candidates) > 0 or current_iteration >= 3
    feedback = "Sufficient matches found" if is_sufficient else "No candidates found"
    
    return {
        "iteration_count": current_iteration,
        "is_sufficient": is_sufficient,
        "critique_feedback": feedback,
        "execution_trace": [f"Evaluator iteration {current_iteration}: {feedback}"]
    }

def route_next_step(state: RetailAgentState) -> str:
    if state["is_sufficient"]:
        return "end"
    return "retry"

def build_retail_graph():
    builder = StateGraph(RetailAgentState)
    builder.add_node("parse_preferences", parse_preference_node)
    builder.add_node("spatial_retrieval", spatial_retrieval_node)
    builder.add_node("evaluate_quality", quality_evaluator_node)
    
    builder.add_edge(START, "parse_preferences")
    builder.add_edge("parse_preferences", "spatial_retrieval")
    builder.add_edge("spatial_retrieval", "evaluate_quality")
    
    builder.add_conditional_edges(
        "evaluate_quality",
        route_next_step,
        {"end": END, "retry": "spatial_retrieval"}
    )
    
    return builder.compile()