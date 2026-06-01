from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class MovieReview(BaseModel):
    title: str = Field(description="Movie title")

    sentiment: Literal[
        "positive",
        "negative",
        "neutral"
    ]

    rating: int = Field(description="Rating from 1 to 10")

    comment: Optional[str] = Field(
        default=None,
        description="Short review comment"
    )


model = ChatOllama(model="phi3")

structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke(
    "Review a sci-fi movie. Include title, sentiment, rating, and comment."
)

print(result)