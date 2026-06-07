from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


model = ChatOllama(model="phi3")

prompt = PromptTemplate(
    template='Write a summary 3 lines for the following topic - \n {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

loader = TextLoader('document_loaders/text.txt', encoding='utf-8')

docs = loader.load()

# print(type(docs))

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'topic':docs[0].page_content}))
