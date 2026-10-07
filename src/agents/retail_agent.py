import operator
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from src.spatial_engine import SpatialDataEngine
from src.rag_engine import RetailRAGEngine

# Instantiate Engines
spatial_engine = SpatialDataEngine()
rag_engine = RetailRAGEngine()

# Seed RAG Engine with sample product catalog
sample_catalog = [
    {"name": "Organic Whole Milk", "category": "Dairy", "description": "Fresh organic milk 1 gallon"},
    {"name": "Almond Milk Unsweetened", "category": "Dairy Alternative", "description": "Plant-based almond milk 64oz"},
    {"name": "Sourdough Bread", "category": "Bakery", "description": "Artisan sourdough loaf"},
    {"name": "Italian Espresso Beans", "category": "Coffee", "description": "Dark roast whole bean coffee"}
]
rag_engine.index_products(sample_catalog)

class RetailAgentState(TypedDict):
    customer_query: str
    extracted_preferences: Dict[str, Any]
    retrieved_locations: List[Dict[str, Any]]
    rag_product_matches: List[Dict[str, Any]]
    critique_feedback: str
    iteration_count: int
    is_sufficient: bool
    execution_trace: Annotated[List[str], operator.add]

def parse_preference_node(state: RetailAgentState) -> Dict[str, Any]:
    return {
        "extracted_preferences": {
            "cuisine": "Italian",
            "max_walk_time_min": 10,
            "ambiance": "casual",
            "item_query": state.get("customer_query", "dairy-free milk alternatives")
        },
        "execution_trace": ["Parsed customer preference constraints"]
    }

def rag_product_search_node(state: RetailAgentState) -> Dict[str, Any]:
    """Node: Queries Qdrant Vector Engine using FastEmbed RAG."""
    query = state["extracted_preferences"].get("item_query", state.get("customer_query", ""))
    products = rag_engine.search_similar_products(query, limit=2)
    
    return {
        "rag_product_matches": products,
        "execution_trace": [f"RAG Engine retrieved {len(products)} relevant catalog items for query: '{query}'"]
    }

def spatial_retrieval_node(state: RetailAgentState) -> Dict[str, Any]:
    """Node: Queries Spatial Engine via Polars."""
    prefs = state["extracted_preferences"]
    
    polars_df = spatial_engine.hybrid_spatial_search(
        target_cuisine=prefs["cuisine"],
        max_walk_time=prefs["max_walk_time_min"]
    )
    
    candidates = polars_df.to_dicts()
    
    return {
        "retrieved_locations": candidates,
        "execution_trace": [f"Spatial Engine retrieved {len(candidates)} spatial matches"]
    }

def quality_evaluator_node(state: RetailAgentState) -> Dict[str, Any]:
    locations = state.get("retrieved_locations", [])
    products = state.get("rag_product_matches", [])
    current_iteration = state.get("iteration_count", 0) + 1
    
    is_sufficient = (len(locations) > 0 or len(products) > 0) or current_iteration >= 3
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
    builder.add_node("rag_search", rag_product_search_node)
    builder.add_node("spatial_retrieval", spatial_retrieval_node)
    builder.add_node("evaluate_quality", quality_evaluator_node)
    
    builder.add_edge(START, "parse_preferences")
    builder.add_edge("parse_preferences", "rag_search")
    builder.add_edge("rag_search", "spatial_retrieval")
    builder.add_edge("spatial_retrieval", "evaluate_quality")
    
    builder.add_conditional_edges(
        "evaluate_quality",
        route_next_step,
        {"end": END, "retry": "rag_search"}
    )
    
    return builder.compile()