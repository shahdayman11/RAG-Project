# 🤖 AI-Powered RAG Coding Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** system designed to help developers and students understand programming code and Python libraries.

The project combines **OCR, semantic search, vector databases, and LLMs** to provide contextual and reliable coding assistance through an interactive **Streamlit** interface.

---

## 🚀 Features

### 👨‍💻 Programming Tutor

The Programming Tutor helps users understand and debug programming code.

**Capabilities:**

* Upload a screenshot of code.
* Extract code from images using **EasyOCR**.
* Validate extracted code using an LLM.
* Ask questions about the uploaded code.
* Retrieve relevant information from the knowledge base.
* Generate contextual explanations and answers.
* Provide beginner-friendly programming guidance.

### 🐍 Python Libraries Assistant

The Python Libraries Assistant provides information and guidance about Python libraries.

**Capabilities:**

* Ask questions about Python libraries.
* Retrieve relevant documentation and knowledge.
* Explain library functions and concepts.
* Provide usage examples.
* Help users understand how and when to use different libraries.
* Generate context-aware answers using RAG.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │      Streamlit      │
                    │    Web Interface    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
      ┌───────────────┐                 ┌─────────────────┐
      │ Programming   │                 │ Python Libraries│
      │ Tutor         │                 │ Assistant       │
      └───────┬───────┘                 └────────┬────────┘
              │                                  │
              ▼                                  ▼
       ┌─────────────┐                    ┌─────────────┐
       │   EasyOCR   │                    │ Knowledge   │
       │ Code Image  │                    │    Base     │
       └──────┬──────┘                    └──────┬──────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Sentence Transformers│
                  │     Embeddings       │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │       FAISS         │
                  │ Semantic Retrieval  │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │      LLM Layer      │
                  │ Gemini / Groq       │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Contextual Answer   │
                  └─────────────────────┘
```

---

## 🔄 RAG Pipeline

The system follows a standard Retrieval-Augmented Generation pipeline:

```text
User Query
    │
    ▼
Query Embedding
    │
    ▼
FAISS Similarity Search
    │
    ▼
Top-K Relevant Documents
    │
    ▼
Context + User Question
    │
    ▼
LLM
    │
    ▼
Generated Answer
```

For the Programming Tutor, an additional OCR pipeline is used:

```text
Code Screenshot
      │
      ▼
   EasyOCR
      │
      ▼
Extracted Code
      │
      ▼
OCR Validation
      │
      ▼
RAG Retrieval
      │
      ▼
Gemini / Groq
      │
      ▼
Programming Explanation
```

---

## 🛠️ Technologies Used

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* Streamlit

### RAG

* FAISS
* Sentence Transformers
* Vector embeddings
* Semantic similarity search

### OCR

* EasyOCR

### LLMs

* Google Gemini
* Groq

### Environment & Utilities

* python-dotenv
* NumPy
* Pickle
* Requests



> Adjust the filenames above to match the final structure of your repository.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
FASTAPI_URL=http://127.0.0.1:8000
```

Never commit your `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
__pycache__/
*.pyc
.venv/
venv/
```

---

## ▶️ Running the Application

### Start FastAPI

```bash
uvicorn main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Start Streamlit

Open another terminal in the same environment:

```bash
streamlit run streamlit_app.py
```

The Streamlit application will then be available in your browser.

---

## 🧠 How RAG Works

Instead of sending the user's question directly to an LLM, the system first searches a knowledge base for relevant information.

### Step 1 — Embedding

Documents and user queries are converted into numerical vectors using a **Sentence Transformer**.

### Step 2 — Retrieval

FAISS performs a similarity search to find the most relevant documents.

### Step 3 — Context Construction

The retrieved documents are combined with the user's question.

### Step 4 — Generation

The combined context is sent to the LLM.

```text
Question
   +
Retrieved Context
   ↓
LLM
   ↓
Context-Aware Response
```

This helps reduce irrelevant answers and allows the assistant to answer using information from the project's knowledge base.

---

## 🖼️ Programming Tutor Workflow

The Programming Tutor supports image-based programming questions.

```text
Screenshot
    ↓
EasyOCR
    ↓
Extract Code
    ↓
Validate / Correct OCR
    ↓
Generate Embedding
    ↓
FAISS Retrieval
    ↓
Retrieve Relevant Context
    ↓
Gemini / Groq
    ↓
Answer
```

For example, a user can upload a screenshot containing Python code and ask:

> "Why am I getting this error?"

The assistant extracts the code, retrieves relevant programming information, and explains the problem.

---

## 🐍 Python Libraries Assistant Workflow

The Python Libraries Assistant focuses on questions such as:

```text
"What does pandas groupby do?"

"How do I use train_test_split?"

"What is the difference between fit() and fit_transform()?"

"How can I create a DataFrame using pandas?"
```

The RAG system retrieves relevant library information before generating the response.

---

## 🔌 API Endpoints

The FastAPI backend exposes endpoints for interacting with the assistants.

Example:

```text
POST /tutor
```

The endpoint can receive programming questions and, when required, uploaded code screenshots.

Additional endpoints can be added for the Python Libraries Assistant depending on the final backend structure.

---

## 🔒 Security

API keys are stored in environment variables and should **never** be hard-coded.

Do not upload:

```text
.env
API keys
private credentials
large model files
```

Use `.gitignore` to prevent accidental commits.

---

## 📌 Future Improvements

* Add conversation memory.
* Support more programming languages.
* Improve OCR code extraction.
* Add syntax highlighting.
* Add code execution in a sandboxed environment.
* Add automatic code debugging.
* Add documentation crawling and automatic knowledge-base updates.
* Add evaluation metrics for retrieval quality.
* Add Docker deployment.
* Deploy the application to a cloud platform.
* Add authentication and user accounts.

---

## 🎯 Project Goal

The goal of this project is to build an intelligent coding assistant that combines **RAG and Large Language Models** with traditional software tools such as OCR and vector search.

It demonstrates the integration of:

**LLMs + RAG + Semantic Search + OCR + FastAPI + Streamlit**

into a practical AI application for programming education and developer assistance.

---

## 👩‍💻 Author

**Nourhan Hamada & Shahd Ayman**

Computer & Data Science Students
Interested in Data Science, Machine Learning, and AI Engineering.

---

## ⭐ Acknowledgements

This project uses open-source technologies and APIs including:

* FastAPI
* Streamlit
* FAISS
* Sentence Transformers
* EasyOCR
* Google Gemini
* Groq
* Python

---

🎥 Demo

Want to see the project in action?

👉 Watch the OCR Demo:-  https://drive.google.com/file/d/1eGkK_qB77yifLyqJok0wq6ZHZnzeQ_YQ/view?usp=sharing

👉 Watch the Python Libraries Assistant Demo:-  https://drive.google.com/file/d/1BU6Ynani_qigY4lpAv9M9EbqN38IYUGZ/view?usp=sharing


The demo shows the Programming Tutor and Python Libraries Assistant, including the RAG retrieval pipeline and Streamlit interface.
