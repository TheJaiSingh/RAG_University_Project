from flask import Flask, render_template, request, jsonify

from dotenv import load_dotenv
import os

import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load Environment Variables

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Flask App

app = Flask(__name__)

# Gemini Model

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# Embedding Model

print("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Database

print("Loading Vector Database...")

db = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("System Ready!")

# Routes

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat")
def chat():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data["question"]

        docs = db.similarity_search(
            question,
            k=1
        )

        unique_docs = []

        seen = set()

        for doc in docs:

            if doc.page_content not in seen:

                unique_docs.append(doc.page_content)

                seen.add(doc.page_content)

        context = "\n\n".join(unique_docs)
        

        sources = list(
            set(
                [
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                    for doc in docs
                ]
            )
        )
        prompt = f"""
You are SRMCEM Smart Campus Assistant.

Answer ONLY from the provided context.

Give short, professional and accurate answers.

If faculty names are available,
list them clearly in bullet points.

If answer is not available, say:

"This information is not available in the university database."

Context:
{context}

Question:
{question}
"""
   

        try:

            response = model.generate_content(
                prompt
            )

            answer = response.text

        except Exception as gemini_error:
            

            print("Gemini Error:", gemini_error)

            answer = "Gemini quota exceeded. Showing PDF data:\n\n" + context


        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:

        return jsonify({
            "answer": f"Error: {str(e)}"
        })

# Run App

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )