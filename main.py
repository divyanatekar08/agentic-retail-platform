from src.agents.retail_agent import build_retail_graph

if __name__ == "__main__":
    app = build_retail_graph()
    
    initial_input = {
        "customer_query": "Looking for a fast casual Italian lunch spot within a 10 min walk in Downtown Brooklyn",
        "iteration_count": 0,
        "execution_trace": []
    }
    
    print("\n--- Executing Stateful Agent Graph ---")
    result = app.invoke(initial_input)
    
    print("\n--- Final Agent State ---")
    print(f"Extracted Preferences: {result['extracted_preferences']}")
    print(f"Retrieved Locations: {result['retrieved_locations']}")
    print(f"Execution Trace: {result['execution_trace']}")