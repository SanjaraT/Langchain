from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatOllama(model="phi3")

# Define output schema
response_schemas = [
    ResponseSchema(
        name="summary",
        description="A 5-line summary of the report"
    ),
    ResponseSchema(
        name="topic",
        description="The topic of the report"
    )
]

parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)

format_instructions = parser.get_format_instructions()

# Prompt
template = PromptTemplate(
    template="""
    Write a detailed report on {topic}.

    Then create a 5-line summary.

    {format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={
        "format_instructions": format_instructions
    }
)

chain = template | model | parser

result = chain.invoke({
    "topic": "black hole"
})

print(result)