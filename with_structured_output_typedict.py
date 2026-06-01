from typing import Literal, Optional, Annotated
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama

class MovieReview(TypedDict):
    title: Annotated[str, "Movie title"]

    sentiment: Literal[
        "positive",
        "negative",
        "neutral"
    ]

    rating: Annotated[int, "Rating from 1 to 10"]

    comment: Optional[str]

model = ChatOllama(model = "phi3")
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("Review a sci-fi movie. Include title, sentiment, rating, and comment.")

print(result)