#import pypdfloader to read pdf files
from langchain_community.document_loaders import PyPDFLoader

#upload pdf files and read them using PyPDFLoader
pdf_files = [  
    "data/college_information.pdf",
    "data/Complete_Fee_Structure.pdf",
    "data/Academic_Calendar.pdf",
    "data/Faculty_information.pdf"
   
    
]
#empty list to store all the documents
all_docs = []

for pdf in pdf_files:
    try:
        print("Loading:", pdf)

        loader = PyPDFLoader(pdf)
        docs = loader.load()

        print("Pages:", len(docs))

        all_docs.extend(docs)

    except Exception as e:
        print("ERROR:", e)

#display first 3 pages
for i in range(3):
    print(f"\n\n===== PAGE {i+1} =====")
    print(all_docs[i].page_content[:500])

#import text splitter 
from langchain_text_splitters import RecursiveCharacterTextSplitter

#create text splitter object
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, #size of each chunk
    chunk_overlap=50 #overlap between chunks
)

#split documents into chunks
chunks = splitter.split_documents(all_docs)

print(f"\nTotal Chunks Created: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("\nCreating Embeddings...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating Vector Database...")

vector_db = FAISS.from_documents(
    chunks,
    embedding_model
)

vector_db.save_local("vector_store")

print("Vector Database Saved Successfully!")

print("\nLoading Vector Database...")

db = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

queries = [
    "MCA fee",
    "Academic Calendar",
    "Placement",
    "Library",
    "MCA Faculty"
]

for query in queries:
    print("\n" + "="*60)
    print("Question:", query)

    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"\n----- Result {i} -----")

        print("Source:",
            doc.metadata.get("source", "Unknown"))

        print(doc.page_content[:300])

for doc in all_docs:
    if "Faculty" in doc.page_content:
        print(doc.page_content)