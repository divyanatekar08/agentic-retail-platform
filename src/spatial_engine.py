import polars as pl
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class SpatialDataEngine:
    def __init__(self):
        # 1. Initialize In-Memory Qdrant Client
        self.qdrant = QdrantClient(":memory:")
        self.collection_name = "downtown_brooklyn_pois"
        
        # 2. Initialize Raw Retail POI Dataset using Polars
        self.raw_poi_df = pl.DataFrame({
            "poi_id": [1, 2, 3, 4],
            "name": ["Fornino Pizza", "Junior's Restaurant", "Han Dynasty", "Custom Fuel Pizza"],
            "cuisine": ["Italian", "American", "Chinese", "Italian"],
            "walk_time_min": [5, 8, 12, 4],
            "rating": [4.6, 4.3, 4.5, 4.1]
        })
        
        self._bootstrap_vector_db()

    def _bootstrap_vector_db(self):
        """Creates collection and inserts mock semantic embeddings (4-dim)."""
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
        
        # Simple mock vectors for testing pipeline
        # (Index 0/3: Italian focus, Index 1: American, Index 2: Asian)
        points = [
            PointStruct(id=1, vector=[0.9, 0.1, 0.0, 0.2], payload={"poi_id": 1, "name": "Fornino Pizza"}),
            PointStruct(id=2, vector=[0.1, 0.8, 0.1, 0.1], payload={"poi_id": 2, "name": "Junior's Restaurant"}),
            PointStruct(id=3, vector=[0.0, 0.1, 0.9, 0.1], payload={"poi_id": 3, "name": "Han Dynasty"}),
            PointStruct(id=4, vector=[0.8, 0.2, 0.0, 0.3], payload={"poi_id": 4, "name": "Custom Fuel Pizza"}),
        ]
        
        self.qdrant.upsert(collection_name=self.collection_name, points=points)

    def hybrid_spatial_search(self, target_cuisine: str, max_walk_time: int) -> pl.DataFrame:
        """Executes Vector Search + Polars Fast In-Memory Spatial Filter."""
        # Query vector representing "Italian Dining"
        query_vector = [0.85, 0.15, 0.0, 0.2]
        
        # Step A: Vector Retrieval using modern Qdrant query_points API
        qdrant_response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=10
        )
        
        matched_ids = [point.payload["poi_id"] for point in qdrant_response.points]
        
        # Step B: Fast Vectorized Polars Filter (Walk Time + Matched IDs)
        filtered_df = self.raw_poi_df.filter(
            (pl.col("poi_id").is_in(matched_ids)) &
            (pl.col("cuisine") == target_cuisine) &
            (pl.col("walk_time_min") <= max_walk_time)
        )
        
        return filtered_df