import os
os.environ["OLLAMA_MODELS"] = r"D:\OllamaModels"


from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# loading model
model = ChatOllama(model= "phi3")

# chat history
chat_history = []

# System instructions
chat_history.append(
    SystemMessage(content="You are a helpful AI assistant!"
    )
)

# loading previous hats
try:
    with open("chat_hidtory.txt", "r",encoding="utf-8")as f:
        lines = f.readlines()
        for line in lines :
            line = line.strip()

            # detect message owner
            if line.startswith("Human:"):
                msg = line.replace("Human:","").strip()
                chat_history.append(HumanMessage(content=msg))

            elif line.startswith("AI:"):
                msg = line.replace("AI:","").strip()
                chat_history.append(AIMessage(content=msg))

except FileNotFoundError:
    pass

print("Chatbot Started!")

# Chat loop
while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        break

    # save human message
    chat_history.append(
        HumanMessage(content=user_input)
    )

    # AI response with the full chat history
    result = model.invoke(chat_history)
    
    # Save AI message
    chat_history.append(
        AIMessage(content=result.content)
    )

    # Print AI response
    print('\nAI:',result.content)
    print()

# Saving the full conversation
with open("chat_history.txt", "a", encoding = "utf-8") as f:
    f.write(f"Human: {user_input}\n")
    f.write(f"AI:{result.content}\n")