# runs multiple steps in oorder, passing each step's output to the next

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model="phi3")

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

print(chain.invoke({'topic': 'My computer science degree certificate'}))