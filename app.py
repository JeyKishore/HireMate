import streamlit as st
from dotenv import load_dotenv

from rag import ResumeRAG
from analyzer import ResumeAnalyzer


# ---------------------------------------
# Load environment variables
# ---------------------------------------

load_dotenv()


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="HireMate",
    page_icon="💼",
    layout="wide"
)


# ---------------------------------------
# Custom CSS
# ---------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .score {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
    }

    .card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# Header
# ---------------------------------------

st.markdown(
    '<div class="main-title">💼 HireMate</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Resume & Job Description Analyzer using RAG'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------
# Initialize RAG
# ---------------------------------------

@st.cache_resource
def load_rag():

    return ResumeRAG()


rag = load_rag()


# ---------------------------------------
# Sidebar
# ---------------------------------------

with st.sidebar:

    st.header("📄 Resume")

    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"]
    )

    st.divider()

    st.info(
        """
        HireMate uses:

        • PDF text extraction
        • Semantic embeddings
        • FAISS vector search
        • Retrieval Augmented Generation
        • LLM-based analysis
        """
    )


# ---------------------------------------
# Main interface
# ---------------------------------------

if uploaded_file is None:

    st.info(
        "👈 Upload your resume PDF to get started."
    )

    st.stop()


# ---------------------------------------
# Process resume
# ---------------------------------------

with st.spinner("Processing resume..."):

    try:

        uploaded_file.seek(0)

        raw_text = rag.extract_text(
            uploaded_file
        )

        cleaned_text = rag.clean_text(
            raw_text
        )

        if not cleaned_text.strip():

            st.error(
                "Could not extract text from this PDF. "
                "Try a text-based PDF instead of a scanned image."
            )

            st.stop()

        chunks = rag.create_chunks(
            cleaned_text
        )

        rag.build_index(chunks)

    except Exception as e:

        st.error(
            f"Error processing resume: {e}"
        )

        st.stop()


st.success(
    f"Resume processed successfully — "
    f"{len(chunks)} chunks created."
)


# ---------------------------------------
# Job Description
# ---------------------------------------

st.subheader("📋 Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=250,
    placeholder="""
Example:

We are looking for an AI/ML Engineer with
experience in Python, Machine Learning,
SQL, NLP and REST APIs...
"""
)


# ---------------------------------------
# Analyze button
# ---------------------------------------

if st.button(
    "🚀 Analyze Resume",
    type="primary",
    use_container_width=True
):

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

        st.stop()

    with st.spinner(
        "Retrieving relevant resume information..."
    ):

        retrieved = rag.retrieve(
            job_description,
            top_k=6
        )

    # -----------------------------------
    # Create context
    # -----------------------------------

    context_parts = []

    for i, item in enumerate(retrieved):

        context_parts.append(
            f"""
RESUME CHUNK {i + 1}
Similarity: {item['score']:.3f}

{item['text']}
"""
        )

    resume_context = "\n".join(
        context_parts
    )

    # -----------------------------------
    # LLM analysis
    # -----------------------------------

    with st.spinner(
        "AI is analyzing your resume..."
    ):

        try:

            analyzer = ResumeAnalyzer()

            result = analyzer.analyze(
                resume_context,
                job_description
            )

        except Exception as e:

            st.error(
                f"LLM error: {e}"
            )

            st.stop()

    # -----------------------------------
    # Match score
    # -----------------------------------

    st.divider()

    st.subheader(
        "📊 Resume–Job Match"
    )

    score = result.get(
        "match_score",
        0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Match Score",
            f"{score}%"
        )

    with col2:

        st.metric(
            "Resume Chunks",
            len(chunks)
        )

    with col3:

        st.metric(
            "Retrieved Chunks",
            len(retrieved)
        )

    # -----------------------------------
    # Summary
    # -----------------------------------

    st.subheader("📝 Summary")

    st.write(
        result.get(
            "summary",
            "No summary available."
        )
    )

    # -----------------------------------
    # Strong skills
    # -----------------------------------

    st.subheader(
        "✅ Strong / Matching Skills"
    )

    skills = result.get(
        "strong_skills",
        []
    )

    if skills:

        cols = st.columns(
            min(len(skills), 4)
        )

        for i, skill in enumerate(skills):

            cols[i % len(cols)].success(
                f"✓ {skill}"
            )

    else:

        st.write(
            "No clearly matching skills found."
        )

    # -----------------------------------
    # Missing skills
    # -----------------------------------

    st.subheader(
        "❌ Missing / Weak Skills"
    )

    missing = result.get(
        "missing_skills",
        []
    )

    if missing:

        for skill in missing:

            st.warning(
                f"• {skill}"
            )

    else:

        st.success(
            "No major missing skills identified."
        )

    # -----------------------------------
    # Projects
    # -----------------------------------

    st.subheader(
        "🚀 Relevant Projects"
    )

    projects = result.get(
        "matching_projects",
        []
    )

    for project in projects:

        st.write(
            f"• {project}"
        )

    # -----------------------------------
    # Experience
    # -----------------------------------

    st.subheader(
        "💼 Experience Analysis"
    )

    st.write(
        result.get(
            "experience_analysis",
            "Not available."
        )
    )

    # -----------------------------------
    # Improvements
    # -----------------------------------

    st.subheader(
        "✍️ Resume Improvements"
    )

    improvements = result.get(
        "resume_improvements",
        []
    )

    for improvement in improvements:

        st.write(
            f"• {improvement}"
        )

    # -----------------------------------
    # Interview questions
    # -----------------------------------

    st.subheader(
        "🎯 Personalized Interview Questions"
    )

    questions = result.get(
        "interview_questions",
        []
    )

    for i, question in enumerate(
        questions,
        start=1
    ):

        with st.expander(
            f"Question {i}"
        ):

            st.write(question)

    # -----------------------------------
    # RAG sources
    # -----------------------------------

    st.divider()

    st.subheader(
        "🔎 Retrieved Resume Context"
    )

    st.caption(
        "These are the resume sections retrieved "
        "using semantic similarity."
    )

    for i, item in enumerate(
        retrieved,
        start=1
    ):

        with st.expander(
            f"Retrieved Chunk {i} "
            f"(Similarity: {item['score']:.3f})"
        ):

            st.write(
                item["text"]
            )