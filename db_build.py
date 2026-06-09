import os

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
import time
from dotenv import load_dotenv
load_dotenv()

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

# Maps FIFA position codes to human-readable labels so semantic search
# correctly matches queries like "striker" or "goalkeeper"
position_labels = {
    "ST": "striker",
    "CF": "striker / centre forward",
    "GK": "goalkeeper",
    "CB": "centre-back",
    "LW": "left winger",
    "RW": "right winger",
    "LM": "left midfielder",
    "RM": "right midfielder",
    "CAM": "attacking midfielder",
    "CDM": "defensive midfielder",
    "CM": "central midfielder",
    "LB": "left back",
    "RB": "right back",
    "LWB": "left wing-back",
    "RWB": "right wing-back",
}

def make_player_text(row):
    # Look up the human-readable label; fall back to the raw code if not found
    position_label = position_labels.get(row['Position'], row['Position'])

    desc = (
        f"{row['Name']} ({row['Age']} years old, {row['Nation']}) plays for {row['Team']} "
        f"in the {row['League']}. Primary position: {row['Position']} ({position_label}). "
        f"Alternative positions: {row['Alternative positions']}. "
        f"Overall rating: {int(row['OVR'])}. "
        f"Play style: {row['play style']}. "
        f"Preferred foot: {row['Preferred foot']}. "
        f"Height: {row['Height']}, Weight: {row['Weight']}.\n\n"

        f"Pace: {int(row['PAC'])}, Acceleration {int(row['Acceleration'])}, "
        f"Sprint Speed {int(row['Sprint Speed'])}.\n"

        f"Shooting: {int(row['SHO'])}, Positioning {int(row['Positioning'])}, "
        f"Finishing {int(row['Finishing'])}, Shot Power {int(row['Shot Power'])}, "
        f"Long Shots {int(row['Long Shots'])}, Volleys {int(row['Volleys'])}, "
        f"Penalties {int(row['Penalties'])}.\n"

        f"Passing: {int(row['PAS'])}, Vision {int(row['Vision'])}, "
        f"Crossing {int(row['Crossing'])}, Free Kick Accuracy {int(row['Free Kick Accuracy'])}, "
        f"Short Passing {int(row['Short Passing'])}, Long Passing {int(row['Long Passing'])}, "
        f"Curve {int(row['Curve'])}.\n"

        f"Dribbling: {int(row['DRI'])}, Agility {int(row['Agility'])}, "
        f"Balance {int(row['Balance'])}, Reactions {int(row['Reactions'])}, "
        f"Ball Control {int(row['Ball Control'])}, Composure {int(row['Composure'])}.\n"

        f"Defending: {int(row['DEF'])}, Interceptions {int(row['Interceptions'])}, "
        f"Heading Accuracy {int(row['Heading Accuracy'])}, Defensive Awareness {int(row['Def Awareness'])}, "
        f"Standing Tackle {int(row['Standing Tackle'])}, Sliding Tackle {int(row['Sliding Tackle'])}.\n"

        f"Physical: {int(row['PHY'])}, Jumping {int(row['Jumping'])}, "
        f"Stamina {int(row['Stamina'])}, Strength {int(row['Strength'])}, "
        f"Aggression {int(row['Aggression'])}.\n"
    )

    # Only add GK attributes for actual goalkeepers or players with nonzero GK stats
    if row['Position'] == "GK" or any(row[[f"GK {stat}" for stat in ["Diving", "Handling", "Kicking", "Positioning", "Reflexes"]]] > 0):
        desc += (
            f"Goalkeeping Attributes: Diving {int(row['GK Diving'])}, "
            f"Handling {int(row['GK Handling'])}, Kicking {int(row['GK Kicking'])}, "
            f"Positioning {int(row['GK Positioning'])}, Reflexes {int(row['GK Reflexes'])}.\n"
        )

    desc += f"Other Details: Weak Foot {row['Weak foot']}-star, Skill Moves {row['Skill moves']}-star."

    return desc

print("Building player text descriptions...")
df["text"] = df.apply(make_player_text, axis=1)

# Load the text column into langchain documents
print("Loading documents...")
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# Strip all metadata — not useful as LLM uses text content for answering questions
for doc in docs:
    doc.metadata = {}

print(f"Embedding {len(docs)} documents — this may take a few minutes...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT_ID"],
    database=os.environ["CHROMA_DATABASE"]
)

# Upload data to Chroma cloud in batches to avoid rate limits (300 docs per minute)
BATCH_SIZE = 250  # safely under the 300 limit

# Create the vectorstore with the first batch
print(f"Uploading batch 1...")
vectorstore = Chroma.from_documents(
    docs[:BATCH_SIZE],
    embedding,
    client=client,
    collection_name="fifa_players"
)

# Add remaining batches
for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    print(f"Uploading batch {i // BATCH_SIZE + 1}...")
    vectorstore.add_documents(batch)
    time.sleep(1)  # small delay to avoid rate limiting

print(f"Done! {len(docs)} players uploaded to Chroma Cloud.")