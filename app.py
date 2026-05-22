from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Embed and store this locally with Chroma
print("Loading vector store...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_fifa_db", embedding_function=embedding)

# Create a retriever, retrieves three most similar players 
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Initilaize local LLM using ollama
print("Loading LLM...")
llm = OllamaLLM(model="phi3:mini")

# RAG chain 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

print("FIFA RAG system up and running. Type 'exit' to quit.")

# Interaction with model
while True:
    query = input("Question: ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print("\nAnswer:\n", result["result"])