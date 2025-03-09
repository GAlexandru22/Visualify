import vertexai
from pymongo import MongoClient
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_google_vertexai import VertexAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
PATH = "./Files"
client = MongoClient(os.getenv('VISUALIFY_DATABASE_URL'))
db = client.get_database('Visualify')
collection = db['BE2']
vertexai.init(project='824838425521',location="us-central1")

def load_doc():
    doc_load = PyPDFDirectoryLoader(PATH)
    return doc_load.load()


def split_doc(documents: list[Document]):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )
    return text_split.split_documents(documents)


def get_embedding_function():
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    return embeddings

def add_to_db(chunks: list[Document]):
    embedding_function = get_embedding_function()

    for chunk in chunks:
        text = chunk.page_content  # Extract text from the chunk
        metadata = chunk.metadata  # Extract metadata (e.g., page number, source file)

        # Generate embedding
        embedding = embedding_function.embed_query(text)

        # Create a document to store in MongoDB
        document = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }

        # Insert into MongoDB
        collection.insert_one(document)
docs = load_doc()
chunks = split_doc(docs)
add_to_db(chunks)
print(chunks[0])

# Connect to the database using the connection URL from environment variables


client.close()

