import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.preprocessing import MinMaxScaler

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

# Normalize numeric values (0–1 range) (commented out for now to see if answers improve)
#scaler = MinMaxScaler()
#df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Combine text
def make_player_text(row):
    desc = (
        f"{row['Name']} ({row['Age']} years old, {row['Nation']}) plays for {row['Team']} "
        f"in the {row['League']}. Primary position: {row['Position']}. "
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

print(f"Embedding {len(docs)} documents — this may take a few minutes...")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding, persist_directory="chroma_fifa_db")

print("Done! Chroma DB saved to ./chroma_fifa_db")