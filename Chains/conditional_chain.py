from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel
)
from pydantic import BaseModel, Field
from typing import Literal

# Local Ollama model
model = ChatOllama(model="phi3")

# Output parsers
str_parser = StrOutputParser()

# Structured output schema
class Feedback(BaseModel):
    sentiment: Literal[
        "positive",
        "negative"
    ] = Field(
        description="Sentiment of the feedback"
    )

# Pydantic parser
feedback_parser = PydanticOutputParser(
    pydantic_object=Feedback
)

# Sentiment classification prompt
classification_prompt = PromptTemplate(
    template="""
Classify the sentiment of the following feedback
as either positive or negative.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions":
        feedback_parser.get_format_instructions()
    }
)

# Classification chain
classifier_chain = (
    classification_prompt
    | model
    | feedback_parser
)

# Keep both sentiment and feedback
prepare_chain = RunnableParallel(
    {
        "feedback": lambda x: x["feedback"],
        "sentiment_result": classifier_chain
    }
)

# Positive response prompt
positive_prompt = PromptTemplate(
    template="""
Write a polite response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Negative response prompt
negative_prompt = PromptTemplate(
    template="""
Write a polite response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

# Branching logic
branch_chain = RunnableBranch(
    (
        lambda x:
        x["sentiment_result"].sentiment == "positive",

        RunnableLambda(
            lambda x: {
                "feedback": x["feedback"]
            }
        )
        | positive_prompt
        | model
        | str_parser
    ),

    (
        lambda x:
        x["sentiment_result"].sentiment == "negative",

        RunnableLambda(
            lambda x: {
                "feedback": x["feedback"]
            }
        )
        | negative_prompt
        | model
        | str_parser
    ),

    RunnableLambda(
        lambda x:
        "Could not determine sentiment."
    )
)

# Complete chain
chain = prepare_chain | branch_chain

# Test
result = chain.invoke(
    {
        "feedback":
        "This is a beautiful phone. I really like its camera."
    }
)

print(result)

