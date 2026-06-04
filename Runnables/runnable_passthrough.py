from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough
)

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

# Generate joke
joke_gen_chain = RunnableSequence(
    prompt1,
    model,
    parser
)

# Create parallel outputs
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(
        prompt2,
        model,
        parser
    )
})

# Full chain
final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({
    'topic': 'my computer science degree certificate'
})

print("JOKE:")
print(result['joke'])

print("\nEXPLANATION:")
print(result['explanation'])