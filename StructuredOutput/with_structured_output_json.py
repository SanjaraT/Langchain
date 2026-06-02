from langchain_ollama import ChatOllama

json_schema = {
    "title": "MovieReview",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Movie title"
        },
        "sentiment": {
            "type": "string",
            "enum": [
                "positive",
                "negative",
                "neutral"
            ]
        },
        "rating": {
            "type": "integer",
            "description": "Rating from 1 to 10"
        },
        "comment": {
            "type": "string",
            "description": "Short review comment"
        }
    },
    "required": [
        "title",
        "sentiment",
        "rating"
    ]
}

model = ChatOllama(model="phi3")

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(
    "Review a sci-fi movie. Include title, sentiment, rating, and comment."
)

print(result)