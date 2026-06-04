# runs multiple tasks simultaneously on the same input & processes it independently, producing a dictionary of outputs

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model="phi3")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(
        prompt1,
        model,
        parser
    ),
    'linkedin': RunnableSequence(
        prompt2,
        model,
        parser
    )
})

result = parallel_chain.invoke({
    'topic': 'AI'
})

print(result['tweet'])
print()
print(result['linkedin'])