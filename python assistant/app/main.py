import os
import pickle

import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from google import genai
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = os.getenv("GEMINI_MODEL")
ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL")


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the .env file."
    )

if not GENERATION_MODEL:
    raise RuntimeError(
        "GEMINI_MODEL was not found in the .env file."
    )

if not ROUTER_MODEL:
    raise RuntimeError(
        "GEMINI_ROUTER_MODEL was not found in the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# PATHS
# ============================================================

VECTOR_STORE_DIR = "vector_store"

FAISS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "python_docs.faiss"
)

METADATA_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "chunk_metadata.pkl"
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(
    FAISS_PATH
)

print(
    "FAISS vectors:",
    index.ntotal
)

print(
    "FAISS dimension:",
    index.d
)


# ============================================================
# LOAD CHUNK METADATA
# ============================================================

print("Loading chunk metadata...")

with open(
    METADATA_PATH,
    "rb"
) as f:

    chunk_metadata = pickle.load(f)


print(
    "Metadata chunks:",
    len(chunk_metadata)
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

print(
    "Loading embedding model:",
    EMBEDDING_MODEL_NAME
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Python Documentation RAG API",
    description=(
        "RAG API using FAISS and Gemini"
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    query: str


# ============================================================
# FAISS SEARCH
# ============================================================

def faiss_search(
    query,
    top_k=5
):

    # Create query embedding

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search FAISS

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if idx < 0:
            continue

        metadata = chunk_metadata[idx]

        # Metadata can have different structures,
        # so we handle the common cases.

        if isinstance(
            metadata,
            dict
        ):

            text = metadata.get(
                "text",
                ""
            )

        else:

            text = str(metadata)

        results.append(
            {
                "rank": rank,
                "index": int(idx),
                "score": float(score),
                "text": text,
                "metadata": metadata
            }
        )

    return results


# ============================================================
# QUERY ROUTER
# ============================================================

def route_query(query):

    prompt = f"""
You are a query router for a Python documentation assistant.

The system has a RAG knowledge base containing documentation
for exactly these libraries:

- NumPy
- Pandas
- Scikit-learn

Classify the user's question into EXACTLY ONE of these
three categories.

RETRIEVE
GENERAL
OUT_OF_CONTEXT

RETRIEVE:
Choose this when the question is specifically about:

- NumPy
- Pandas
- Scikit-learn
- Their functions
- Their classes
- Their APIs
- Their usage
- Their parameters
- Their behavior
- Their examples
- Their errors
- Their documentation

GENERAL:
Choose this when the question is about Python itself
or general programming concepts, but is NOT specifically
about NumPy, Pandas, or Scikit-learn.

OUT_OF_CONTEXT:
Choose this when the question is unrelated to:

- Python
- Programming
- NumPy
- Pandas
- Scikit-learn

IMPORTANT:

A question about NumPy, Pandas, or Scikit-learn MUST be
RETRIEVE.

A general Python question is GENERAL.

An unrelated question is OUT_OF_CONTEXT.

Return ONLY ONE of:

RETRIEVE
GENERAL
OUT_OF_CONTEXT

User question:

{query}
"""

    response = client.models.generate_content(
        model=ROUTER_MODEL,
        contents=prompt
    )

    route = response.text.strip().upper()

    valid_routes = {
        "RETRIEVE",
        "GENERAL",
        "OUT_OF_CONTEXT"
    }

    if route not in valid_routes:

        route = "OUT_OF_CONTEXT"

    return route


# ============================================================
# GENERAL PYTHON ANSWER
# ============================================================

PYTHON_DOCS_URL = (
    "https://docs.python.org/3/"
)


def general_answer(query):

    return (
        "This is a general Python question.\n\n"
        "Official Python documentation:\n"
        f"{PYTHON_DOCS_URL}"
    )


# ============================================================
# RAG ANSWER
# ============================================================

def rag_answer(
    query,
    context
):

    prompt = f"""
You are a Python documentation assistant.

Your knowledge source for this answer is ONLY the
documentation provided below.

The documentation comes from:

- NumPy
- Pandas
- Scikit-learn

Answer the user's question using ONLY the provided context.

Do NOT use outside knowledge.

Do NOT invent information.

If the provided context does not contain enough information
to answer the question, say:

"I don't know based on the provided documentation."

Keep the answer clear and concise.

============================================================
DOCUMENTATION CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{query}

============================================================
ANSWER
============================================================
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
DOCUMENT CHUNK {result['rank']}

{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# MAIN QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query_endpoint(
    request: QueryRequest
):

    query = request.query.strip()

    if not query:

        return {
            "error": "Query cannot be empty."
        }


    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    route = route_query(
        query
    )


    # --------------------------------------------------------
    # OUT OF CONTEXT
    # --------------------------------------------------------

    if route == "OUT_OF_CONTEXT":

        return {
            "query": query,
            "route": route,
            "answer": "This question is out of context.",
            "sources": []
        }


    # --------------------------------------------------------
    # GENERAL PYTHON
    # --------------------------------------------------------

    if route == "GENERAL":

        return {
            "query": query,
            "route": route,
            "answer": general_answer(
                query
            ),
            "sources": []
        }


    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = faiss_search(
        query,
        top_k=5
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    answer = rag_answer(
        query,
        context
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    for result in results:

        metadata = result[
            "metadata"
        ]

        if isinstance(
            metadata,
            dict
        ):

            sources.append(
                {
                    "rank": result[
                        "rank"
                    ],

                    "score": result[
                        "score"
                    ],

                    "source": metadata.get(
                        "source"
                    ),

                    "page": metadata.get(
                        "page"
                    ),

                    "library": metadata.get(
                        "library"
                    )
                }
            )

        else:

            sources.append(
                {
                    "rank": result[
                        "rank"
                    ],

                    "score": result[
                        "score"
                    ],

                    "source": None,

                    "page": None,

                    "library": None
                }
            )


    return {
        "query": query,
        "route": route,
        "answer": answer,
        "sources": sources
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "Python Documentation RAG API is running."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "faiss_vectors": index.ntotal,
        "embedding_dimension": index.d
    }