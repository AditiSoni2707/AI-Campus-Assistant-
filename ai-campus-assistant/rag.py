from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load campus documents
files = [
    "data/faq.txt",
    "data/rules.txt",
    "data/timetable.txt"
]

documents = []
for file in files:
    loader = TextLoader(file)
    documents.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

# Prompt
prompt = ChatPromptTemplate.from_template(
    """You are a campus assistant.
Answer the question using ONLY the context below.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}
"""
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain 
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("LEVEL 2: Campus RAG Assistant (type exit to quit)")

while True:
    question = input("Student: ")
    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)
    print("Assistant:", answer)


