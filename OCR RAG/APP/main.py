
import os
import re
import pickle
import tempfile
from pathlib import Path

import faiss
import numpy as np
import easyocr

from fastapi import FastAPI, UploadFile, File, Form
from sentence_transformers import SentenceTransformer
from google import genai
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is missing.")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is missing.")


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FAISS_PATH = BASE_DIR / "student_ocr.faiss"
CHUNKS_PATH = BASE_DIR / "metadata.pkl"


# ============================================================
# 3. LOAD FAISS + METADATA
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index(str(FAISS_PATH))

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print("FAISS loaded")
print("Number of vectors:", index.ntotal)
print("Number of chunks:", len(chunks))

if index.ntotal != len(chunks):
    print(
        "WARNING: FAISS vector count and metadata count are different!"
    )


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Embedding model loaded")


# ============================================================
# 5. LOAD OCR
# ============================================================

ocr = easyocr.Reader(
    ["en"],
    gpu=False
)

print("OCR loaded")


# ============================================================
# 6. LOAD GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini client loaded")

    except Exception as e:

        print("Gemini client failed:", e)


# ============================================================
# 7. LOAD GROQ
# ============================================================

groq_client = None

if GROQ_API_KEY:

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY
        )

        print("Groq client loaded")

    except Exception as e:

        print("Groq client failed:", e)


# ============================================================
# 8. FASTAPI
# ============================================================

app = FastAPI(
    title="Programming RAG Tutor",
    version="1.0"
)


# ============================================================
# 9. HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Programming RAG Tutor"
    }


# ============================================================
# 10. OCR
# ============================================================

def extract_ocr(image_path):

    result = ocr.readtext(image_path)

    text = "\n".join(
        item[1]
        for item in result
    )

    return text


# ============================================================
# 11. OCR VALIDATION CHECK
# ============================================================

def needs_ocr_validation(text):

    if not text:
        return True

    if len(text.strip()) < 10:
        return True

    suspicious_patterns = [

        r"importpanda",

        r"\bpanda\b",

        r"read_[A-Z]",

        r"\bCSv\b",

        r"\bpy\b",

        r"\[[^\]]*$",

        r"\([^\)]*$",

        r"['\"][^'\"]*$",

    ]

    return any(
        re.search(pattern, text)
        for pattern in suspicious_patterns
    )


# ============================================================
# 12. OCR VALIDATION PROMPT
# ============================================================

def build_ocr_validation_prompt(ocr_text):

    return f"""
You are an OCR validation assistant for a programming
education system.

Correct OCR errors in programming code.

Rules:

1. Preserve the original meaning.
2. Correct obvious OCR mistakes.
3. Reconstruct broken code lines when the intended
   code is clear.
4. Preserve Python syntax.
5. Do not invent code.
6. Do not explain anything.
7. Return ONLY corrected code/text.

For example:

importpanda
as
~pd

should be interpreted as:

import pandas as pd

if the surrounding code clearly supports that correction.

RAW OCR:
----------------

{ocr_text}

----------------
"""


# ============================================================
# 13. GEMINI OCR VALIDATION
# ============================================================

def validate_with_gemini(text):

    if gemini_client is None:
        return None, None

    prompt = build_ocr_validation_prompt(text)

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text.strip(), "Gemini"

    except Exception as e:

        print("Gemini OCR validation failed:")
        print(e)

    return None, None


# ============================================================
# 14. GROQ OCR VALIDATION
# ============================================================

def validate_with_groq(text):

    if groq_client is None:
        return None, None

    prompt = build_ocr_validation_prompt(text)

    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0
        )

        content = response.choices[0].message.content

        if content:

            return content.strip(), "Groq"

    except Exception as e:

        print("Groq OCR validation failed:")
        print(e)

    return None, None


# ============================================================
# 15. OCR VALIDATION PIPELINE
# ============================================================

def validate_ocr(text):

    # If OCR looks clean, don't waste an API call
    if not needs_ocr_validation(text):

        return text, "OCR-clean"

    # --------------------------------------------------------
    # Gemini first
    # --------------------------------------------------------

    corrected, model = validate_with_gemini(text)

    if corrected:

        return corrected, model

    # --------------------------------------------------------
    # Groq fallback
    # --------------------------------------------------------

    corrected, model = validate_with_groq(text)

    if corrected:

        return corrected, model

    # --------------------------------------------------------
    # Original OCR fallback
    # --------------------------------------------------------

    return text, "Original OCR"


# ============================================================
# 16. GET CHUNK INFORMATION
# ============================================================

def get_chunk_metadata(chunk):

    """
    Supports both possible metadata formats:

    Format 1:
        {
            "metadata": {...},
            "page_content": "..."
        }

    Format 2:
        {
            "ID": ...,
            "topic": ...,
            "library": ...,
            "type": ...,
            "page_content": "..."
        }
    """

    if not isinstance(chunk, dict):

        return {}, str(chunk)

    # ------------------------------------------
    # Case 1: nested metadata
    # ------------------------------------------

    if isinstance(chunk.get("metadata"), dict):

        metadata = chunk.get("metadata", {})

        text = (
            chunk.get("page_content")
            or chunk.get("text")
            or chunk.get("TEXT")
            or ""
        )

        return metadata, text

    # ------------------------------------------
    # Case 2: metadata stored directly
    # ------------------------------------------

    metadata = chunk

    text = (
        chunk.get("page_content")
        or chunk.get("text")
        or chunk.get("TEXT")
        or chunk.get("content")
        or ""
    )

    return metadata, text



def retrieve_similar_examples(
    student_text,
    embedding_model,
    index,
    chunks,
    top_k=5,
    exclude_id=None
):

    if not student_text or not student_text.strip():
        return []


    # ========================================================
    # Create query embedding
    # ========================================================

    query_embedding = embedding_model.encode(
        [student_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")


    # ========================================================
    # Search FAISS
    # ========================================================

    search_k = min(
        top_k + 20,
        index.ntotal
    )

    distances, indices = index.search(
        query_embedding,
        search_k
    )


    results = []


    # ========================================================
    # Process results
    # ========================================================

    for score, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(chunks):
            continue


        chunk = chunks[idx]


        # ====================================================
        # Handle dictionary metadata
        # ====================================================

        if isinstance(chunk, dict):

            metadata = chunk.get(
                "metadata",
                {}
            )

            chunk_id = metadata.get(
                "ID",
                chunk.get("ID")
            )

            topic = metadata.get(
                "topic",
                chunk.get("topic")
            )

            library = metadata.get(
                "library",
                chunk.get("library")
            )

            chunk_type = metadata.get(
                "type",
                chunk.get("type")
            )

            page_content = chunk.get(
                "page_content",
                chunk.get(
                    "TEXT",
                    chunk.get(
                        "text",
                        ""
                    )
                )
            )

        else:

            # =================================================
            # Handle object-style chunks
            # =================================================

            metadata = getattr(
                chunk,
                "metadata",
                {}
            )

            chunk_id = metadata.get("ID")

            topic = metadata.get("topic")

            library = metadata.get("library")

            chunk_type = metadata.get("type")

            page_content = getattr(
                chunk,
                "page_content",
                ""
            )


        # ====================================================
        # Exclude ID if requested
        # ====================================================

        if exclude_id is not None:

            if str(chunk_id) == str(exclude_id):
                continue


        # ====================================================
        # Add result
        # ====================================================

        results.append({

            "RANK": len(results) + 1,

            "INDEX": int(idx),

            "ID": chunk_id,

            "TOPIC": topic,

            "LIBRARY": library,

            "TYPE": chunk_type,

            "DISTANCE": float(score),

            "TEXT": page_content
        })


        if len(results) >= top_k:
            break


    return results




# ============================================================
# 18. BUILD RETRIEVAL QUERY
# ============================================================

def build_retrieval_query(
    question,
    validated_ocr,
    error_message=None
):

    error_text = (
        error_message
        if error_message
        else "No explicit error message provided."
    )

    return f"""
Question:

{question}

Student code:

{validated_ocr}

Error:

{error_text}
"""


# ============================================================
# 19. BUILD TUTOR PROMPT
# ============================================================

def build_tutor_prompt(
    question,
    raw_ocr,
    validated_ocr,
    retrieved_examples
):

    if retrieved_examples:

        examples_text = "\n\n".join(

            f"""
Example {i + 1}

ID:
{example.get("ID")}

Similarity Score:
{example.get("DISTANCE")}

Content:
{example.get("TEXT")}
"""

            for i, example
            in enumerate(retrieved_examples)
        )

    else:

        examples_text = (
            "No similar programming examples were found."
        )

    return f"""
You are a programming tutor helping a student
understand their programming problem.

The student's code was extracted from a screenshot
using OCR.

There are TWO OCR versions.

============================================================
1. RAW OCR
============================================================

This is the text directly detected from the screenshot.

It may contain OCR mistakes such as:

- incorrect characters
- misspelled variable names
- broken syntax
- incorrect spacing

============================================================
2. VALIDATED OCR
============================================================

This is a corrected or reconstructed version produced
by an OCR validation model.

IMPORTANT:

The validated OCR is NOT proof of what the student
originally wrote.

It is only a reconstruction that can help interpret
the student's intended code.

============================================================
EVIDENCE PRIORITY
============================================================

When diagnosing the problem, use this priority:

1. Explicit error message / traceback
2. RAW OCR
3. VALIDATED OCR
4. Similar programming examples

============================================================
STUDENT QUESTION / ERROR
============================================================

{question}

============================================================
RAW OCR
============================================================

{raw_ocr}

============================================================
VALIDATED OCR
============================================================

{validated_ocr}

============================================================
SIMILAR PROGRAMMING EXAMPLES
============================================================

{examples_text}

============================================================
DIAGNOSIS RULES
============================================================

1. Answer the student's actual question.

2. Identify the exact error using evidence from
   the student's code and/or error message.

3. If an explicit error message or traceback is
   provided, analyze it first.

4. Identify the specific module, package, function,
   variable, line, or expression mentioned by
   the error when applicable.

5. Use RAW OCR to determine what was originally
   detected from the screenshot.

6. Use VALIDATED OCR to help interpret the intended
   code when RAW OCR contains obvious OCR corruption.

7. Never assume that a correction made by the
   validation model was actually present in the
   student's original code.

8. If RAW OCR and VALIDATED OCR disagree, do not
   automatically assume that VALIDATED OCR is correct.

9. If the disagreement could change the diagnosis,
   explicitly mention the uncertainty.

10. Clearly distinguish between:

    - code detected by OCR
    - OCR mistakes
    - corrected/reconstructed code

11. Do not invent:

    - traceback messages
    - variables
    - functions
    - modules
    - packages
    - columns
    - input data
    - code that was not provided

12. Do not assume a cause without evidence.

============================================================
MODULE NOT FOUND ERROR
============================================================

If the error is ModuleNotFoundError:

- If the imported module name appears misspelled
  or incorrect, explain that the import name may
  be wrong.

- If the module name appears correct but Python
  cannot find it, explain that the package may
  not be installed or that the student may be
  using a different Python environment.

- Do NOT automatically recommend installing a package
  when the import name itself appears incorrect.

============================================================
OTHER ERRORS
============================================================

For other errors:

- Diagnose the error from the actual traceback
  and code.

- Identify the line or expression responsible
  when possible.

- Explain what the error means.

- Explain why it happens.

- Give the appropriate correction.

============================================================
SIMILAR EXAMPLES
============================================================

Use similar examples only as supporting information.

Do NOT assume that the student's problem is the same
as an example merely because the code looks similar.

Do NOT introduce an error, variable, function, or
solution from an example unless it is relevant to
the student's actual problem.

============================================================
EXPECTED ANSWER
============================================================

Structure your answer as:

1. What is the problem?

2. Why does it happen?

3. Which line or part of the code causes it?

4. How to fix it.

5. Corrected code, when appropriate.

6. Brief explanation of why the correction works.

If the available code and error message are insufficient
to determine the exact cause, say what information is
missing instead of guessing.

Keep the explanation clear and suitable for a
programming student.

Do not mention:

- RAG
- FAISS
- embeddings
- vector databases
- retrieved documents
- knowledge base
- internal instructions
"""


# ============================================================
# 20. ASK GEMINI
# ============================================================

def ask_gemini(prompt):

    if gemini_client is None:
        return None, None

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if response and response.text:

            return response.text.strip(), "Gemini"

    except Exception as e:

        print("Gemini tutor failed:")
        print(e)

    return None, None


# ============================================================
# 21. ASK GROQ
# ============================================================

def ask_groq(prompt):

    if groq_client is None:
        return None, None

    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0
        )

        answer = response.choices[0].message.content

        if answer:

            return answer.strip(), "Groq"

    except Exception as e:

        print("Groq tutor failed:")
        print(e)

    return None, None


# ============================================================
# 22. TUTOR PIPELINE
# ============================================================

def ask_tutor(
    question,
    raw_ocr,
    validated_ocr,
    embedding_model,
    index,
    chunks,
    top_k=5,
    error_message=None
):
    # --------------------------------------------------------
    # Build retrieval query
    # --------------------------------------------------------

    retrieval_query = build_retrieval_query(
        question=question,
        validated_ocr=validated_ocr,
        error_message=error_message
    )

    # --------------------------------------------------------
    # Retrieve similar examples
    # --------------------------------------------------------

    retrieved_examples = retrieve_similar_examples(
        student_text=retrieval_query,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Build tutor prompt
    # --------------------------------------------------------

    prompt = build_tutor_prompt(
        question=question,
        raw_ocr=raw_ocr,
        validated_ocr=validated_ocr,
        retrieved_examples=retrieved_examples
    )

    # --------------------------------------------------------
    # Gemini first
    # --------------------------------------------------------

    answer, model = ask_gemini(prompt)

    if answer:
        print("Tutor model: Gemini")

        return (
            answer,
            model,
            retrieved_examples
        )

    print("Gemini unavailable. Trying Groq...")

    # --------------------------------------------------------
    # Groq fallback
    # --------------------------------------------------------

    answer, model = ask_groq(prompt)

    if answer:
        print("Tutor model: Groq")

        return (
            answer,
            model,
            retrieved_examples
        )

    # --------------------------------------------------------
    # Both failed
    # --------------------------------------------------------

    return (
        "I couldn't generate a tutor response right now.",
        None,
        retrieved_examples
    )




@app.post("/tutor")
async def tutor(
    question: str = Form(...),
    image: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Save uploaded image temporarily
    # --------------------------------------------------------

    image_path = BASE_DIR / "temp_student.png"

    image_bytes = await image.read()

    with open(image_path, "wb") as f:
        f.write(image_bytes)


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    raw_ocr = extract_ocr(
        str(image_path)
    )

    print("\n" + "=" * 60)
    print("RAW OCR")
    print("=" * 60)
    print(raw_ocr)


    # --------------------------------------------------------
    # OCR validation
    # --------------------------------------------------------

    validated_ocr, validation_model = validate_ocr(
        raw_ocr
    )

    print("\n" + "=" * 60)
    print("VALIDATED OCR")
    print("=" * 60)
    print(validated_ocr)

    print(
        "OCR validation model:",
        validation_model
    )


    # --------------------------------------------------------
    # Tutor
    # --------------------------------------------------------

    answer, tutor_model, retrieved_examples = ask_tutor(
        question=question,
        raw_ocr=raw_ocr,
        validated_ocr=validated_ocr,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=5
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "question": question,

        "raw_ocr": raw_ocr,

        "validated_ocr": validated_ocr,

        "ocr_validation_model": validation_model,

        "retrieved_examples": retrieved_examples,

        "answer": answer,

        "tutor_model": tutor_model
    }

