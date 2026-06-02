from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="phi3")
parser = StrOutputParser()

# Step 1: Generate report
report_prompt = PromptTemplate(
    template="Write a short report about {topic}",
    input_variables=["topic"]
)

# Step 2: Summarize report
summary_prompt = PromptTemplate(
    template="Summarize the following report in 3 lines:\n\n{text}",
    input_variables=["text"]
)

# Step 3: Generate title
title_prompt = PromptTemplate(
    template="Generate a catchy title for this summary:\n\n{summary}",
    input_variables=["summary"]
)

# Sequential chain
report_chain = report_prompt | model | parser
summary_chain = summary_prompt | model | parser
title_chain = title_prompt | model | parser

# Execute sequentially
report = report_chain.invoke({"topic": "Artificial Intelligence"})

summary = summary_chain.invoke({
    "text": report
})

title = title_chain.invoke({
    "summary": summary
})

print("REPORT:")
print(report)

print("\nSUMMARY:")
print(summary)

print("\nTITLE:")
print(title)