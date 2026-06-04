# convert any python function into  runnables

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model="phi3")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

result = final_chain.invoke({
    'topic': 'AI'
})

final_result = "{}\n\nWord Count - {}".format(
    result['joke'],
    result['word_count']
)

print(final_result)