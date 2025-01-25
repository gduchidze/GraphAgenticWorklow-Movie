import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import unicodedata

def to_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

load_dotenv(".env")

df = pd.read_csv("imdb_top_1000.csv")

openai_api_key = os.environ.get("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
index = pc.Index("movies-database")

for _, row in df.iterrows():
    text = f"""
    Title: {row['Series_Title']}
    Year: {row['Released_Year']}
    Runtime: {row.get('Runtime', 'N/A')}
    Genre: {row.get('Genre', 'N/A')}
    IMDB Rating: {row.get('IMDB_Rating', 'N/A')}
    Overview: {row.get('Overview', 'No description available')}
    Meta Score: {row.get('Meta_score', 'N/A')}
    Director: {row.get('Director', 'N/A')}
    """

    vector = embeddings.embed_query(text)

    metadata = {
        "text": text,
        "title": row["Series_Title"],
        "year": str(row["Released_Year"]),
        "certificate": row.get("Certificate", "N/A"),
        "runtime": row.get("Runtime", "N/A"),
        "genre": row.get("Genre", "N/A"),
        "rating": str(row.get("IMDB_Rating", "N/A")),
        "meta_score": str(row.get("Meta_score", "N/A")),
        "director": row.get("Director", "N/A"),
        "stars": f"{row.get('Star1', 'N/A')}, {row.get('Star2', 'N/A')}, {row.get('Star3', 'N/A')}, {row.get('Star4', 'N/A')}",
        "votes": str(row.get("No_of_Votes", "N/A")),
        "gross": str(row.get("Gross", "N/A"))
    }

    for key, value in metadata.items():
        if pd.isna(value) or value is None:
            metadata[key] = "N/A"

    vector = embeddings.embed_query(text)
    vector_id = to_ascii(str(row["Series_Title"]))
    index.upsert([(vector_id, vector, metadata)])

print("✅")
