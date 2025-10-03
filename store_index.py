from src.helper import load_pdf_file, text_splitter, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_splitter(extracted_data)
pc = Pinecone(api_key="pcsk_7MgPhr_NUGymS6R54idzTjgZP5UnU3V3nXA8Ngq9s1mwpgqoKgPm9puFGF3K2FktnPwUe8")

index_name = "medical-chatbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
         cloud="aws",
         region="us-east-1",
        )
)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)