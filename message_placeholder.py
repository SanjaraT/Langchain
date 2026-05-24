# Message placeholder is used inside a ChatPromptTemplate to dynamically insert chat history or a list of messages at runtime.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama

# Create chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Initialize Ollama model
model = ChatOllama(model="phi3")

chat_history = []

# Load chat history
with open('chat_history.txt', 'r') as f:
    lines = f.readlines()

# Convert text lines into message objects
for line in lines:
    if line.startswith("Human:"):
        chat_history.append(
            HumanMessage(content=line.replace("Human:", "").strip())
        )

    elif line.startswith("AI:"):
        chat_history.append(
            AIMessage(content=line.replace("AI:", "").strip())
        )

print(chat_history)

# Create prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund?'
})

# Invoke model
result = model.invoke(prompt)

print(result.content)