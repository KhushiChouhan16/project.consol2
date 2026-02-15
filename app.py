import streamlit as st
import os
<<<<<<< HEAD
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Code Reviewer", layout="wide")

# -------------------------------
# Subtle UI Styling
# -------------------------------
st.markdown("""
<style>
    .main {
        background-color: #EEEEEE;
    }

    h2, h3 {
        color: #213C51;
        font-weight: 600;
    }

    .subtle-text {
        color: #6594B1;
        font-size: 0.95rem;
    }

    .divider {
        height: 1px;
        background-color: #DDAED3;
        margin: 1.6rem 0;
    }

    .footer {
        color: #6594B1;
        font-size: 1.05rem;
        text-align: center;
        margin-top: 2.5rem;
    }

    .download-section {
        margin-top: 2.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------------
# Header (NOT DULL, SAME AS DIVIDER)
# -------------------------------
st.markdown("""
<h1 style="
    color:#DDAED3;
    font-size:2.6rem;
    font-weight:700;
    letter-spacing:0.4px;
    margin-bottom:0.3rem;
">
    AI-Powered Code Reviewer
</h1>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='subtle-text'>Review Python code for bugs, optimizations, readability, and best practices.</p>",
    unsafe_allow_html=True
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# -------------------------------
# Example Code
# -------------------------------
EXAMPLE_CODE = """def largest(arr):
    max = 0
    for i in arr:
        if i > max:
            max = i
    return max
"""

# -------------------------------
# Session State Initialization
# -------------------------------
if "code_input" not in st.session_state:
    st.session_state.code_input = ""

# -------------------------------
# Code Input Section
# -------------------------------
st.subheader("Code Input")
st.markdown("<p class='subtle-text'>Paste your Python code below or load an example.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])

with col1:
    if st.button("Load Example Code"):
        st.session_state.code_input = EXAMPLE_CODE

with col2:
    st.write("")

code_input = st.text_area(
    "Python Code",
    height=280,
    value=st.session_state.code_input,
    placeholder="def add(a, b):\n    return a + b"
)

# -------------------------------
# Review Button
# -------------------------------
if st.button("Review Code"):
    if code_input.strip() == "":
        st.warning("Please paste some Python code first.")
    else:
        with st.spinner("Reviewing your code..."):
            try:
                prompt = f"""
You are a senior Python developer and code reviewer.

Review the following Python code and provide:
1. Bugs or logical errors
2. Optimization suggestions
3. Readability and style improvements
4. Improved version of the code with comments

Also give a final Code Quality Score out of 10 and a one-line justification.
Format it clearly as:
Score: X/10
Reason: <one line>

Python Code:
{code_input}
"""

                response = client.models.generate_content(
                    model="models/gemini-flash-latest",
                    contents=prompt
                )

                full_text = response.text

                # -------------------------------
                # Extract Score
                # -------------------------------
                score = None
                if "Score:" in full_text:
                    try:
                        score_part = full_text.split("Score:")[1]
                        score = score_part.split("/")[0].strip()
                    except:
                        score = None

                # -------------------------------
                # Split Sections
                # -------------------------------
                bugs_text = "No bugs section found."
                opt_text = "No optimization section found."
                code_text = "No improved code section found."

                try:
                    bugs_text = full_text.split("1. Bugs")[1].split("2. Optimization")[0]
                except:
                    pass

                try:
                    opt_text = full_text.split("2. Optimization")[1].split("3. Readability")[0]
                except:
                    pass

                try:
                    code_text = full_text.split("Improved Version")[1]
                except:
                    pass

                # -------------------------------
                # Score
                # -------------------------------
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.subheader("Code Quality Score")

                if score:
                    st.metric("Overall Score", f"{score} / 10")
                else:
                    st.write("Score not found")

                # -------------------------------
                # Review Tabs
                # -------------------------------
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.subheader("Detailed Review")

                tab1, tab2, tab3 = st.tabs([
                    "Bugs & Issues",
                    "Optimizations",
                    "Improved Code"
                ])

                with tab1:
                    st.markdown(bugs_text)

                with tab2:
                    st.markdown(opt_text)

                with tab3:
                    st.markdown(code_text)

                # -------------------------------
                # Download Section
                # -------------------------------
                report_text = f"""
AI CODE REVIEW REPORT
====================

Code Quality Score: {score if score else 'N/A'} / 10

--------------------
BUGS & ISSUES
--------------------
{bugs_text}

--------------------
OPTIMIZATIONS
--------------------
{opt_text}

--------------------
IMPROVED CODE
--------------------
{code_text}
"""

                st.markdown("<div class='download-section'>", unsafe_allow_html=True)
                st.subheader("Download Results")

                st.download_button(
                    label="Download Full Review Report",
                    data=report_text,
                    file_name="ai_code_review_report.txt",
                    mime="text/plain"
                )

                st.download_button(
                    label="Download Improved Code",
                    data=code_text,
                    file_name="improved_code.py",
                    mime="text/plain"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            except ClientError as e:
                st.error("API quota issue. Please try again later.")
                st.code(str(e))

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<div class='divider'></div>"
    "<div class='footer'>Built with ❤️ by <strong>Khushi</strong> · Python · Streamlit · Gemini API</div>",
    unsafe_allow_html=True
)
=======
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="Intelligent PDF Chatbot (RAG)",
    layout="wide"
)

st.title("📄 Intelligent PDF Chatbot (RAG)")
st.caption("Ask questions strictly from the uploaded PDF")

# ================= SESSION STATE =================
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# ================= LAYOUT =================
left_col, right_col = st.columns([1, 2])

# ================= LEFT PANEL (UPLOAD) =================
with left_col:
    st.subheader("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        # Reset state ONLY if new PDF uploaded
        if st.session_state.current_pdf != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.session_state.current_pdf = uploaded_file.name

        VECTOR_DIR = f"vectorstore_{uuid.uuid4().hex}"
        os.makedirs(VECTOR_DIR, exist_ok=True)
        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join("uploads", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ PDF uploaded successfully")
        st.write(f"📄 **{uploaded_file.name}**")

        # -------- LOAD PDF --------
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        st.info(f"Pages loaded: {len(documents)}")

        # -------- SPLIT TEXT --------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        st.info(f"Chunks created: {len(chunks)}")

        if len(chunks) == 0:
            st.error("❌ No readable text found in this PDF.")
            st.stop()

        # -------- EMBEDDINGS + VECTOR DB --------
        with st.spinner("Creating vector database..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=VECTOR_DIR
            )

        st.success("✅ Vector database ready")

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # -------- LLM (OLLAMA) --------
        llm = Ollama(model="gemma3:4b")

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

# ================= RIGHT PANEL (CHAT) =================
with right_col:
    st.subheader("💬 Chat with your PDF")

    if st.session_state.qa_chain is None:
        st.info("Upload a PDF to start chatting.")
    else:
        # ---------- FIXED QUERY BAR (TOP) ----------
        user_question = st.chat_input("Ask a question from the PDF")

        # ---------- HANDLE QUESTION ----------
        if user_question:
            with st.spinner("Searching document..."):
                answer = st.session_state.qa_chain.run(user_question)

            # Insert latest Q&A at TOP
            st.session_state.chat_history.insert(
                0, (user_question, answer)
            )

        # ---------- SHOW LATEST ANSWER ----------
        if st.session_state.chat_history:
            latest_q, latest_a = st.session_state.chat_history[0]

            with st.chat_message("user"):
                st.markdown(latest_q)

            with st.chat_message("assistant"):
                st.markdown(latest_a)

        # ---------- SHOW OLD HISTORY ----------
        if len(st.session_state.chat_history) > 1:
            st.divider()
            st.markdown("### 📜 Previous Questions")

            for q, a in st.session_state.chat_history[1:]:
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    st.markdown(a)

# ================= FOOTER =================
st.divider()
st.caption("Built with LangChain • ChromaDB • Ollama • Streamlit")
st.caption("Project by Khushi Chouhan")
>>>>>>> f66dd2684838d7e6b770f47a4f73d3aa28341809
