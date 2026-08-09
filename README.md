# HireMate
HireMate is an AI- Resume analyzer

# 💼 HireMate — AI-Powered Resume Analyzer

> **An AI-powered Resume & Job Description Analyzer built using Retrieval-Augmented Generation (RAG), semantic search, FAISS, Sentence Transformers, and Gemini.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![RAG](https://img.shields.io/badge/AI-RAG-purple.svg)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-orange.svg)](https://ai.google.dev/)

---

## 🚀 Overview

**HireMate** is a Retrieval-Augmented Generation (RAG) based application designed to help students and job seekers understand how well their resume matches a specific job description.

Instead of relying solely on an LLM's general knowledge, HireMate retrieves relevant information directly from the candidate's resume using **semantic vector search** and provides that retrieved context to the Gemini LLM for grounded analysis.

The application can identify relevant skills, missing skills, suitable projects, resume improvement areas, and personalized technical interview questions.

---

## 🎯 Problem Statement

Job seekers often struggle to understand:

* How well their resume matches a particular job description
* Which required skills are missing
* Which projects are relevant to the role
* How to improve their resume
* What technical questions they may face during an interview

Traditional keyword-based resume matching can fail when similar concepts are expressed using different words.

For example:

```text
"Machine Learning Engineer"
```

and

```text
"ML Developer"
```

may be semantically related even though they do not contain exactly the same keywords.

HireMate addresses this problem using **semantic embeddings and vector similarity search**.

---

## 💡 Solution

HireMate processes a resume through the following pipeline:

```text
Resume PDF
    │
    ▼
Text Extraction
    │
    ▼
Text Cleaning
    │
    ▼
Chunking
    │
    ▼
Sentence Transformer
    │
    ▼
Vector Embeddings
    │
    ▼
FAISS Vector Database
    │
    │
    ├────────────── Job Description
    │                       │
    │                       ▼
    │                 Query Embedding
    │                       │
    │                       ▼
    └──────────────► Similarity Search
                            │
                            ▼
                     Top-K Resume Chunks
                            │
                            ▼
                         Gemini
                            │
                            ▼
                     AI Analysis
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          Skills       Improvements   Interview
          Analysis                    Questions
```

---

# ✨ Features

### 📄 1. Resume Upload

Upload a resume in PDF format.

HireMate extracts the text using **PyMuPDF** and prepares it for semantic retrieval.

---

### 🎯 2. Resume–Job Matching

Paste a job description and receive an AI-generated assessment of the resume's relevance to the role.

---

### ✅ 3. Matching Skills

Identifies skills from the resume that are relevant to the job description.

Example:

```text
Python
Machine Learning
SQL
Computer Vision
Flask
```

---

### ❌ 4. Missing / Weak Skills

Identifies important job requirements that are not clearly present in the retrieved resume content.

Example:

```text
Docker
AWS
NLP
REST APIs
```

---

### 🚀 5. Relevant Projects

Identifies projects from the resume that are relevant to the target job.

---

### ✍️ 6. Resume Improvement Suggestions

Provides practical suggestions to improve the resume for the selected role.

---

### 🎤 7. Personalized Interview Questions

Generates technical interview questions based on:

* Candidate's resume
* Candidate's projects
* Candidate's skills
* Target job description

Example:

```text
1. Explain the architecture of your EcoVert project.

2. Why did you choose your machine learning algorithm?

3. How does cosine similarity work?

4. Explain the RAG pipeline used in HireMate.

5. How would you deploy your ML model as a REST API?
```

---

### 🔎 8. Semantic Retrieval

HireMate uses vector embeddings and FAISS to retrieve the most relevant sections of the resume.

This allows semantic matching instead of relying purely on exact keyword matching.

---

# 🧠 RAG Implementation

HireMate follows the standard Retrieval-Augmented Generation architecture.

## Step 1 — Document Loading

The resume PDF is loaded using **PyMuPDF**.

```python
document = fitz.open(
    stream=pdf_file.read(),
    filetype="pdf"
)
```

---

## Step 2 — Text Extraction

Text is extracted from each PDF page and combined into a single document.

---

## Step 3 — Chunking

The extracted resume text is divided into smaller chunks.

```text
Resume
   ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

Chunking allows the retrieval system to search smaller pieces of information instead of processing the entire resume every time.

---

## Step 4 — Embedding Generation

Each chunk is converted into a numerical vector using:

**`all-MiniLM-L6-v2`**

These vectors represent the semantic meaning of the text.

Conceptually:

```text
"Machine Learning Engineer"
          ↓
[0.12, -0.42, 0.71, ...]
```

---

## Step 5 — Vector Storage

The generated embeddings are stored in **FAISS**.

FAISS is used for efficient similarity search over high-dimensional vectors.

---

## Step 6 — Query Embedding

The job description is also converted into an embedding.

```text
Job Description
       ↓
Embedding
       ↓
FAISS Search
```

---

## Step 7 — Similarity Search

FAISS retrieves the most relevant resume chunks.

The implementation normalizes the embeddings and uses inner-product search, which is equivalent to cosine similarity for normalized vectors.

```text
Query
  ↓
Embedding
  ↓
FAISS
  ↓
Top-K Relevant Chunks
```

---

## Step 8 — Retrieval-Augmented Generation

The retrieved resume chunks are provided as context to the Gemini model.

```text
Retrieved Resume Context
          +
Job Description
          ↓
        Gemini
          ↓
     AI Analysis
```

This grounds the generated response in the candidate's resume rather than asking the LLM to invent candidate information.

---

# 🛠️ Technology Stack

| Technology                | Purpose                         |
| ------------------------- | ------------------------------- |
| **Python**                | Core programming language       |
| **Streamlit**             | Web application interface       |
| **PyMuPDF**               | PDF text extraction             |
| **Sentence Transformers** | Text embeddings                 |
| **FAISS**                 | Vector similarity search        |
| **Gemini API**            | Large Language Model            |
| **NumPy**                 | Numerical operations            |
| **python-dotenv**         | Environment variable management |

---

# 📁 Project Structure

```text
HireMate/
│
├── app.py
├── rag.py
├── analyzer.py
├── requirements.txt
│
├── .env.example
├── .gitignore
├── README.md
│
├── data/
│   └── .gitkeep
│
└── screenshots/
    ├── dashboard.png
    ├── analysis.png
    └── interview_questions.png
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/HireMate.git
```

Move into the project directory:

```bash
cd HireMate
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate:

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_available_gemini_model
```

You can use `.env.example` as a template.

### ⚠️ Security

**Never commit your `.env` file or API key to GitHub.**

The repository includes `.gitignore` to prevent accidental commits.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Or, on Windows when using the project virtual environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🖥️ Application Workflow

### Step 1

Upload your resume:

```text
Resume.pdf
```

### Step 2

Paste the target job description.

### Step 3

Click:

```text
Analyze Resume
```

### Step 4

HireMate performs:

```text
Resume Processing
       ↓
Embedding Generation
       ↓
Vector Search
       ↓
Relevant Context Retrieval
       ↓
Gemini Analysis
```

### Step 5

The application displays:

* Resume–job relevance
* Matching skills
* Missing skills
* Relevant projects
* Experience analysis
* Resume improvements
* Personalized interview questions
* Retrieved resume context

---

# 📊 Example Use Case

### Input

**Resume:**

```text
Python
Machine Learning
SQL
Computer Vision
Flask
Power BI

Projects:
EcoVert
CampusBuddy
Gestro
```

### Job Description

```text
Looking for an AI/ML Engineer with experience in:

Python
Machine Learning
Deep Learning
Computer Vision
NLP
SQL
REST APIs
Docker
AWS
```

### HireMate Analysis

```text
Strong Skills:
✓ Python
✓ Machine Learning
✓ SQL
✓ Computer Vision

Potential Gaps:
• NLP
• Docker
• AWS

Relevant Projects:
• EcoVert
• CampusBuddy

Recommendations:
• Add REST API experience
• Highlight measurable ML results
• Mention deployment technologies
```

---

# 🎤 Interview Preparation

One of the main goals of HireMate is to help candidates prepare for technical interviews.

The system generates questions based on the candidate's actual resume and target role.

For example:

```text
Explain your EcoVert project.

Why did you select the machine learning algorithm
used in your project?

How does FAISS perform similarity search?

What is the difference between RAG and fine-tuning?

How would you deploy an ML model using Flask?
```

---

# 🧪 Technical Concepts Demonstrated

This project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* Large Language Models
* Natural Language Processing
* Text embeddings
* Semantic search
* Vector databases
* FAISS
* Cosine similarity
* Document chunking
* Prompt engineering
* Structured LLM output
* Python
* Streamlit
* API integration

---

# ⚖️ RAG vs Fine-Tuning

| RAG                            | Fine-Tuning                               |
| ------------------------------ | ----------------------------------------- |
| Retrieves external information | Trains the model on additional data       |
| Doesn't modify model weights   | Modifies model weights                    |
| Easy to update knowledge       | Requires retraining for knowledge changes |
| Useful for private documents   | Useful for specialized behavior           |
| Used by HireMate               | Not required for this project             |

---

# 🔒 Privacy & Security

HireMate is designed to avoid storing API credentials in source code.

### Credentials

API keys are loaded through environment variables / deployment secrets.

### Resume Data

Resume files are processed for analysis and should not be committed to the repository.

Users should avoid uploading highly sensitive personal documents to untrusted deployments.

---

# ⚠️ Current Limitations

The current version is an MVP.

Known limitations include:

* Resume parsing works best with text-based PDFs.
* Complex layouts and scanned PDFs may require OCR.
* Chunking is currently based on text length rather than document structure.
* Semantic similarity does not guarantee perfect skill matching.
* LLM-generated assessments should be treated as recommendations rather than definitive hiring decisions.
* The current retrieval pipeline is optimized for relatively small document collections.

---

# 🚀 Future Enhancements

Planned improvements include:

* [ ] Multi-resume comparison
* [ ] Multiple job description comparison
* [ ] Hybrid keyword + semantic search
* [ ] Reranking models
* [ ] OCR support for scanned resumes
* [ ] Better section-aware chunking
* [ ] Transparent semantic scoring
* [ ] Resume rewriting
* [ ] Cover letter generation
* [ ] Interview answer evaluation
* [ ] Persistent vector database
* [ ] User authentication
* [ ] Resume version tracking
* [ ] Analytics dashboard
* [ ] Cloud deployment
* [ ] Automated RAG evaluation

---

# 🌐 Deployment

HireMate can be deployed using **Streamlit Community Cloud**.

Deployment flow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Configure Secrets
       ↓
Deploy
       ↓
Public Web Application
```

For deployment, configure:

```text
GEMINI_API_KEY
GEMINI_MODEL
```

using the platform's secret management system.

---

# 📸 Screenshots

Add your application screenshots here after deployment.

### Dashboard

```text
screenshots/dashboard.png
```

### Resume Analysis

```text
screenshots/analysis.png
```

### Interview Questions

```text
screenshots/interview_questions.png
```

---

# 📌 Why I Built This

I built HireMate as a practical application of **Retrieval-Augmented Generation** to solve a real-world problem encountered during placement preparation.

The project helped me gain hands-on experience in:

> **Semantic Search → Vector Retrieval → Context Augmentation → LLM Generation**

rather than simply calling an LLM API.

---

# 👨‍💻 Author

**JEY KISHORE K**

B.Tech — Artificial Intelligence & Data Science

Interested in:

* Artificial Intelligence
* Machine Learning
* Generative AI
* Data Science
* Full-Stack Development

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and portfolio purposes.

---

