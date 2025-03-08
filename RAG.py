from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
import vertexai
from google.cloud import language_v1
import API

api_key_string = API.create_api_key(824838425521, "abs")

def authenticate_with_api_key(api_key_string: str) -> None:
    """
    Authenticates with an API key for Google Language service.

    TODO(Developer): Replace this variable before running the sample.

    Args:
        api_key_string: The API key to authenticate to the service.
    """

    # Initialize the Language Service client and set the API key
    client = language_v1.LanguageServiceClient(
        client_options={"api_key": api_key_string}
    )

    text = "Hello, world!"
    document = language_v1.Document(
        content=text, type_=language_v1.Document.Type.PLAIN_TEXT
    )

    # Make a request to analyze the sentiment of the text.
    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    print(f"Text: {text}")
    print(f"Sentiment: {sentiment.score}, {sentiment.magnitude}")
    print("Successfully authenticated using the API key")

# Create a RAG Corpus, Import Files, and Generate a response

# TODO(developer): Update and un-comment below lines
PROJECT_ID = "824838425521"
display_name = "Visualify"
paths = ["https://drive.google.com/drive/u/0/folders/1U9luXRNAPg6YSCsIqE7GIXaOyTtgclas"]  # Supports Google Cloud Storage and Google Drive Links

# Initialize Vertex AI API once per session
vertexai.init(project=PROJECT_ID, location="us-central1")

# Create RagCorpus
# Configure embedding model, for example "text-embedding-004".
embedding_model_config = rag.EmbeddingModelConfig(
    publisher_model="publishers/google/models/text-embedding-004"
)

rag_corpus = rag.create_corpus(
    display_name=display_name,
    embedding_model_config=embedding_model_config,
)

# Import Files to the RagCorpus
rag.import_files(
    rag_corpus.name,
    paths,
    chunk_size=512,  # Optional
    chunk_overlap=100,  # Optional
    max_embedding_requests_per_min=900,  # Optional
)

# Direct context retrieval
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
            # Optional: supply IDs from `rag.list_files()`.
            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
        )
    ],
    text="What is RAG and why it is helpful?",
    similarity_top_k=10,  # Optional
    vector_distance_threshold=0.5,  # Optional
)
print(response)

# Enhance generation
# Create a RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            similarity_top_k=3,  # Optional
            vector_distance_threshold=0.5,  # Optional
        ),
    )
)
# Create a gemini-pro model instance
rag_model = GenerativeModel(
    model_name="gemini-2.0-flash", tools=[rag_retrieval_tool]
)

# Generate response
response = rag_model.generate_content("Explain to me what is the concept of electromagnetic field from the files given")
print(response.text)
