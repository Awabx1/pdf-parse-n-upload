import os
import time
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# create index if it does not exist, otherwise populate the existing index

# customize parameters here
cloud='aws'
regiion='us-east-1'
index_name = "custom_index_name"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
print(existing_indexes)
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=regiion),
    )

# Wait until the index is ready
while True:
    index_description = pc.describe_index(index_name)
    status = index_description.get("status", {})
    state = status.get("state")
    if state != "Ready":
        time.sleep(5)
    else:
        break

index = pc.Index(index_name)

def load_and_parse_pdf(pdf_path):
    #Returns parsed text
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

def upload_to_pinecone(document_text):
    # make up chunks of the parsed text and upload using add_texts function 

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document_text)
    
    # upload texts to Pinecone vector store
    vector_store.add_texts(chunks)

    print("Data uploaded to Pinecone")

if __name__ == "__main__":
    pdf_path = ""  # Replace with the actual path to your PDF file
    try:
        text = load_and_parse_pdf(pdf_path)
        print('PDF Parsed')
        upload_to_pinecone(text)
    except Exception as e:
        print(f"Error parsing PDF: {e}")