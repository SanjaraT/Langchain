from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

model = ChatOllama(model="phi3")

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()
url = 'https://www.daraz.com.bd/products/3-cbh-145-1-1-20-i323265911-s1539597147.html?scm=1007.51610.379274.0&pvid=520e4cbd-a473-4c76-b0fe-1f6a07da20ba&search=flashsale&spm=a2a0e.tm80335411.FlashSale.d_323265911'
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(chain.invoke({'question':'What is the product that we are talking about?', 'text':docs[0].page_content}))