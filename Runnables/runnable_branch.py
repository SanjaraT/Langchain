# for conditional chains

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch


# Prompts 
prompt1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summarize the following text\n{text}',
    input_variables=['text']
    )

model= ChatOllama(model = "phi3")
parser = StrOutputParser()

# report 
report_chain = prompt1 | model | parser

# if report is more than 300 words, summarize it 
branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300,
        prompt2 | model | parser
    ),
    # else keep it as it is
    RunnablePassthrough()
)

# final chain
final_chain = RunnableSequence(
    report_chain, branch_chain
)


result = final_chain.invoke({
    'topic':'Palestine'
})

print(result)