from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Schema
class MovieReview(BaseModel):
    title : str = Field(description = "Movie title")
    sentiment : str = Field(description="Positive, Negative or Neutral")
    rating : str = Field(description = "Rating from 1 to 10")
    comment : str = Field(description = "Short review comment")

# Parser
parser = PydanticOutputParser(pydantic_object= MovieReview)

# model 
model = ChatOllama(model = "phi3")

# prompt 
prompt = PromptTemplate(
    template = """
    Review a sci-fi movie
    {format_instructions}
    """,
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser
result = chain.invoke({})
print(result) 