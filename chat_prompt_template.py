# list of messages(multi-turn convo) --> Dynamic messages(ChatPromttemplate) 

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Create template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# Create prompt
prompt = chat_template.invoke({
    'domain': 'medical',
    'topic': 'diabetes'
})

# Load Ollama model
model = ChatOllama(model='phi3')

# Invoke model
result = model.invoke(prompt)

print(result.content)