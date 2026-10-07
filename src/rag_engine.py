import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

class RetailRAGEngine:
    def __init__(self, collection_name: str = "retail_catalog"):
        self.collection_name = collection_name
        
        # Initialize fast, local embedding model (BAAI/bge-small-en-v1.5 -> 384 dimensions)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize in-memory Qdrant client
        self.qdrant = QdrantClient(":memory:")
        self._initialize_collection()

    def _initialize_collection(self):
        # bge-small-en-v1.5 output vector size is 384
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def _get_embedding(self, text: str) -> List[float]:
        # FastEmbed returns a generator of numpy arrays
        embeddings = list(self.embedding_model.embed([text]))
        return embeddings[0].tolist()

    def index_products(self, products: List[Dict[str, Any]]):
        """Indexes product catalog dictionaries into Qdrant."""
        points = []
        for idx, item in enumerate(products):
            text_to_embed = f"{item.get('name', '')} {item.get('category', '')} {item.get('description', '')}"
            vector = self._get_embedding(text_to_embed)
            
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=vector,
                    payload=item
                )
            )
            
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_similar_products(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Performs vector similarity search against indexed products."""
        query_vector = self._get_embedding(query)
        
        # Modern Qdrant API method
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return [hit.payload for hit in response.points]