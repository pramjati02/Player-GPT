import pandas as pd 
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data 
df = pd.read_csv("all_players.csv")

# List of numeric columns
numeric_cols = [
    "OVR", "PAC", "SHO", "PAS", "DRI", "DEF", "PHY",
    "Acceleration", "Sprint Speed", "Positioning", "Finishing", "Shot Power", "Long Shots",
    "Volleys", "Penalties", "Vision", "Crossing", "Free Kick Accuracy", "Short Passing",
    "Long Passing", "Curve", "Dribbling", "Agility", "Balance", "Reactions", "Ball Control",
    "Composure", "Interceptions", "Heading Accuracy", "Def Awareness", "Standing Tackle",
    "Sliding Tackle", "Jumping", "Stamina", "Strength", "Aggression",
    "GK Diving", "GK Handling", "GK Kicking", "GK Positioning", "GK Reflexes"
]

# Replace missing numeric values with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# Normalize numeric values (0–1 range)
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Combine text — skip GK stats if irrelevant
def make_player_text(row):
    desc = (
        f"{row['Name']} ({row['Age']} years old, {row['Nation']}) plays for {row['Team']} "
        f"in the {row['League']}. Primary position: {row['Position']}. "
        f"Alternative positions: {row['Alternative positions']}. "
        f"Overall rating: {int(row['OVR'] * 100)}. "
        f"Play style: {row['play style']}. "
        f"Preferred foot: {row['Preferred foot']}. "
        f"Height: {row['Height']}, Weight: {row['Weight']}.\n\n"

        f"Pace: {int(row['PAC'] * 100)}, Acceleration {int(row['Acceleration'] * 100)}, "
        f"Sprint Speed {int(row['Sprint Speed'] * 100)}.\n"

        f"Shooting: {int(row['SHO'] * 100)}, Positioning {int(row['Positioning'] * 100)}, "
        f"Finishing {int(row['Finishing'] * 100)}, Shot Power {int(row['Shot Power'] * 100)}, "
        f"Long Shots {int(row['Long Shots'] * 100)}, Volleys {int(row['Volleys'] * 100)}, "
        f"Penalties {int(row['Penalties'] * 100)}.\n"

        f"Passing: {int(row['PAS'] * 100)}, Vision {int(row['Vision'] * 100)}, "
        f"Crossing {int(row['Crossing'] * 100)}, Free Kick Accuracy {int(row['Free Kick Accuracy'] * 100)}, "
        f"Short Passing {int(row['Short Passing'] * 100)}, Long Passing {int(row['Long Passing'] * 100)}, "
        f"Curve {int(row['Curve'] * 100)}.\n"

        f"Dribbling: {int(row['DRI'] * 100)}, Agility {int(row['Agility'] * 100)}, "
        f"Balance {int(row['Balance'] * 100)}, Reactions {int(row['Reactions'] * 100)}, "
        f"Ball Control {int(row['Ball Control'] * 100)}, Composure {int(row['Composure'] * 100)}.\n"

        f"Defending: {int(row['DEF'] * 100)}, Interceptions {int(row['Interceptions'] * 100)}, "
        f"Heading Accuracy {int(row['Heading Accuracy'] * 100)}, Defensive Awareness {int(row['Def Awareness'] * 100)}, "
        f"Standing Tackle {int(row['Standing Tackle'] * 100)}, Sliding Tackle {int(row['Sliding Tackle'] * 100)}.\n"

        f"Physical: {int(row['PHY'] * 100)}, Jumping {int(row['Jumping'] * 100)}, "
        f"Stamina {int(row['Stamina'] * 100)}, Strength {int(row['Strength'] * 100)}, "
        f"Aggression {int(row['Aggression'] * 100)}.\n"
    )

    # Add GK attributes only if the player is a GK or has nonzero GK stats
    if row['Position'] == "GK" or any(row[[f"GK {stat}" for stat in ["Diving", "Handling", "Kicking", "Positioning", "Reflexes"]]] > 0):
        desc += (
            f"Goalkeeping Attributes: Diving {int(row['GK Diving'] * 100)}, "
            f"Handling {int(row['GK Handling'] * 100)}, Kicking {int(row['GK Kicking'] * 100)}, "
            f"Positioning {int(row['GK Positioning'] * 100)}, Reflexes {int(row['GK Reflexes'] * 100)}.\n"
        )

    desc += f"Other Details: Weak Foot {row['Weak foot']}-star, Skill Moves {row['Skill moves']}-star."

    return desc

df["text"] = df.apply(make_player_text, axis=1)

# Load the text column into langchain documents
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# Embed and store this locally with Chroma
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding, persist_directory="chroma_fifa_db")

# Create a retriever, retrieves three most similar players 
retriever = vectorstore.as_retriever(search_kwargs={"k":6}) 

# Initilaize local LLM using ollama
llm = OllamaLLM(model="phi3:mini")

# RAG chain 
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# Interaction with model
print("FIFA RAG system up and running. Type 'exit' to quit program")

while True:
    query = input("Question: ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print("\n Answer:\n", result["result"])
    #print("\n Context used\n")
    #for doc in result["source_documents"]:
    #       print("-", doc.page_content)
    #print("\n" + "-" * 80)
