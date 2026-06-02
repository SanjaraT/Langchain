import os

os.environ["OLLAMA_MODELS"] = r"D:\OllamaModels"

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

model = Ollama(model="phi3")

template2 = PromptTemplate(
    template='Greet this person in 5 languages. The name of the person is {name}',
    input_variables=['name']
)

prompt = template2.invoke({'name':'Sanjara'})

result = model.invoke(prompt)

print(result)
