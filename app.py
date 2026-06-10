import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic import hub
from langchain_groq import ChatGroq
import chromadb
from dotenv import load_dotenv
load_dotenv()
 
app = FastAPI()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
 
class Query(BaseModel):
    input: str

# retrieve text format in which LLM processes context and query 
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# API keys for retrieving vector store from Chroma cloud
client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT_ID"],
    database=os.environ["CHROMA_DATABASE"]
)

# Retrieve vector store from Chroma cloud, initialize the retrviever, and load LLM
print("Loading vector store...")
vectorstore = Chroma(client=client, collection_name="fifa_players")
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

print("Loading LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant")

# This tells the LLM what the question is and the context from the retriever, formatted in a way that the LLM can understand
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

# The retrieval chain which combines the Chroma retriever and the LLM formatted text 
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
 
print("Ready.")


# Serve static files (styles.css, script.js)
app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
 
@app.post("/ask")
def ask(query: Query):
    result = retrieval_chain.invoke({"input": query.input})
    return {"response": result["answer"]}
