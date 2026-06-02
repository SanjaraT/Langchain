from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model="phi3")

# JSON parser
parser = JsonOutputParser()

# First prompt
template1 = PromptTemplate(
    template="""
    Write a detailed report on {topic}.

    Return the response in JSON format:
    {{
        "report": "your detailed report"
    }}
    """,
    input_variables=["topic"]
)

# Second prompt
template2 = PromptTemplate(
    template="""
    Write a 5-line summary of the following report.

    Report:
    {report}

    Return the response in JSON format:
    {{
        "summary": "your summary"
    }}
    """,
    input_variables=["report"]
)

# Step 1: Generate report
report_chain = template1 | model | parser

report_result = report_chain.invoke({
    "topic": "black hole"
})

print("Report:")
print(report_result)

# Step 2: Generate summary
summary_chain = template2 | model | parser

summary_result = summary_chain.invoke({
    "report": report_result["report"]
})

print("\nSummary:")
print(summary_result)