import dspy

class ProductCategorizer(dspy.Signature):
    """Classify retail products and extract key attributes for catalog management."""

    product_description: str = dspy.InputField(desc="Raw product title or description")
    category: str = dspy.OutputField(desc="Standard retail taxonomy category")
    tags: list[str] = dspy.OutputField(desc="Keywords or search tags")
    confidence_score: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")


class CatalogAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.categorize = dspy.ChainOfThought(ProductCategorizer)

    def forward(self, product_description: str):
        return self.categorize(product_description=product_description)