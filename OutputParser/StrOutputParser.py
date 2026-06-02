from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="phi3")

# 1st prompt
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

parser = StrOutputParser()

# The chain takes a topic, generates a detailed report about it using Phi-3, then feeds that report back into the model to produce a concise 5-line summary, returning the final result as plain text.
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)