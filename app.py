import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import chromadb
from dotenv import load_dotenv
load_dotenv()
 
app = FastAPI()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
 
class Query(BaseModel):
    input: str
 
print("Loading vector store...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT_ID"],
    database=os.environ["CHROMA_DATABASE"]
)

vectorstore = Chroma(client=client, collection_name="fifa_players", embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
 
print("Loading LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant")
 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)
 
print("Ready.")
 
# Serve static files (styles.css, script.js)
app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
 
@app.post("/ask")
def ask(query: Query):
    result = qa_chain.invoke({"query": query.input})
    return {"response": result["result"]}