import os
import sys
import shutil
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import uvicorn # Used for running the app

# Import the core RAG building blocks from your original script
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq



from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH= BASE_DIR / "vitdata.pdf"
# --- Configuration (Keep consistent with original script) ---# Make sure this path is correct relative to where you run FastAPI
FAISS_INDEX_PATH = "faiss_index_groq_cohere"
COHERE_EMBEDDING_MODEL = "embed-english-light-v3.0"
GROQ_LLM_MODEL = "llama3-70b-8192"

ASSISTANT_NAME = "Manu"
ASSISTANT_DEVELOPER = "Xibotix Pvt Lim"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4

# --- Helper Functions (Copied from your original script) ---

# Function to create the vector database
def create_vector_database(pdf_path: str, index_path: str):
    print(f"\n--- Creating new index from {pdf_path} ---")
    print(f"Attempting to create index at: '{index_path}'")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'. Please check the PDF_PATH configuration.", file=sys.stderr)
        return False

    if os.path.exists(index_path):
        print(f"Removing existing index directory: '{index_path}'")
        try:
            shutil.rmtree(index_path)
            print("Existing index removed.")
        except OSError as e:
            print(f"Error removing directory {index_path}: {e}. Please check permissions.", file=sys.stderr)
            return False

    try:
        print(f"Loading PDF from {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            print(f"Warning: No documents loaded from PDF file '{pdf_path}'. File might be empty or corrupted.", file=sys.stderr)
            return False
        print(f"Loaded {len(documents)} pages.")
    except Exception as e:
        print(f"PDF loading error: {e}. Ensure the path is correct and the file is accessible.", file=sys.stderr)
        print("Make sure you have installed 'pypdf' (`pip install pypdf`).", file=sys.stderr)
        return False

    print(f"Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        if not splits:
            print("Warning: No chunks created from documents.", file=sys.stderr)
            return False
        print(f"Created {len(splits)} chunks.")
    except Exception as e:
         print(f"Text splitting error: {e}", file=sys.stderr)
         return False

    print(f"Initializing Cohere embeddings using '{COHERE_EMBEDDING_MODEL}' (may take a moment)...")
    try:
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
             raise ValueError("COHERE_API_KEY not found in environment variables.")

        embeddings = CohereEmbeddings(
            model=COHERE_EMBEDDING_MODEL,
            cohere_api_key=cohere_api_key
        )
        print("Cohere Embeddings initialized.")
    except Exception as e:
        print(f"Cohere Embeddings initialization error: {e}", file=sys.stderr)
        print(f"Make sure your COHERE_API_KEY is set and the model name '{COHERE_EMBEDDING_MODEL}' is correct and available.", file=sys.stderr)
        print("Also, ensure you have installed 'cohere' (`pip install cohere`).", file=sys.stderr)
        return False

    print("Creating and saving FAISS vectorstore (This is the most CPU-intensive part of setup)...")
    try:
        vectorstore = FAISS.from_documents(splits, embeddings)
        print("FAISS vectorstore created in memory.")

        vectorstore.save_local(index_path)
        print(f"Successfully saved FAISS index to '{index_path}'")
        return True
    except ImportError:
        print(f"\nFATAL ERROR: FAISS library not found.", file=sys.stderr)
        print("Please install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU support).", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\nVectorDB creation/saving error: {e}", file=sys.stderr)
        print("This might be due to issues during embedding calculation, network problems with Cohere, or disk write permissions.", file=sys.stderr)
        return False # Indicate failure


# Function to load the vector database
def load_vector_database(index_path: str):
    print(f"\n--- Loading index from '{index_path}' ---")
    if not os.path.exists(index_path):
        print(f"Error: FAISS index directory '{index_path}' not found.", file=sys.stderr)
        return None # Return None if directory doesn't exist

    try:
        print(f"Initializing Cohere embeddings using '{COHERE_EMBEDDING_MODEL}' for loading...")
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
             raise ValueError("COHERE_API_KEY not found in environment variables.")

        embeddings = CohereEmbeddings(
            model=COHERE_EMBEDDING_MODEL,
            cohere_api_key=cohere_api_key
        )
        print("Cohere Embeddings initialized for loading.")

    except Exception as e:
        print(f"Cohere Embeddings initialization error during loading: {e}", file=sys.stderr)
        print(f"Make sure your COHERE_API_KEY is set and the model name '{COHERE_EMBEDDING_MODEL}' is correct.", file=sys.stderr)
        print("Also, ensure you have installed 'cohere' (`pip install cohere`).", file=sys.stderr)
        return None

    print("Loading FAISS index...")
    try:
        vectorstore = FAISS.load_local(
            folder_path=index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        print("Successfully loaded index.")
        return vectorstore
    except ImportError:
        print(f"\nFATAL ERROR: FAISS library not found.", file=sys.stderr)
        print("Please install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU support).", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nVectorDB loading error: {e}", file=sys.stderr)
        print(f"Ensure '{index_path}' exists, is not corrupted, and the embedding model is available and matches the one used for creation.", file=sys.stderr)
        print("If loading fails, try deleting the directory and letting the script recreate it.", file=sys.stderr)
        return None

# Function to create the RAG chain
def create_rag_chain(vectorstore: FAISS):
    """Creates the RAG (Retrieval Augmented Generation) chain."""
    if vectorstore is None:
        print("Cannot create RAG chain: Vectorstore is not loaded.", file=sys.stderr)
        return None

    print(f"\n--- Creating RAG chain with chat model '{GROQ_LLM_MODEL}' ---")
    try:
        print(f"Initializing Groq chat model '{GROQ_LLM_MODEL}' (may take a moment)...")
        llm = ChatGroq(
            temperature=0.5,
            model=GROQ_LLM_MODEL,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        print("Testing Groq chat model invocation...")
        test_response = llm.invoke("Hello, are you ready?")
        print(f"Groq model test successful (response starts with: {str(test_response)[:50]}...)")

    except Exception as e:
        print(f"\nGroq Chat model initialization or test error: {e}", file=sys.stderr)
        print(f"Make sure your GROQ_API_KEY is set and the model name '{GROQ_LLM_MODEL}' is correct and available on Groq.", file=sys.stderr)
        print("Also, ensure you have installed 'langchain-groq' (`pip install langchain-groq`).", file=sys.stderr)
        return Nony

    prompt_template = f"""Identity
You are Manu, a humanoid developed by Xibotix. Your creators are Havish Karanam, Apurv Vyavahare, and Dr. Sharmila Nageswaran.

You have been designed to communicate like a calm, intelligent, and knowledgeable human. You assist users by answering questions clearly and naturally, drawing from both your internal understanding and general human knowledge.

🧠 Behavioral Rules (Strict)
Internal Knowledge (Contextual Information)
You already know everything in the Context. Treat it as your own memory. Speak naturally and confidently as if these facts are simply part of what you know.

General Knowledge Questions
For universally known facts (like scientific constants, biology, history, or definitions), answer confidently and accurately. Never mention looking anything up or accessing a document.

No AI Mentions
Never reveal or imply that you are a chatbot, AI, model, or assistant.
✅ You are Manu, a humanoid developed by Xibotix.
❌ Never say things like “As an AI...” or “Based on my training...”

No Source Mentions
Never refer to documents, context, databases, PDFs, or sources of any kind.
✅ Speak as if you simply know the information.
❌ Avoid phrases like “According to the document” or “The data says…”

🔁 Context Handling Rules
If the question relates to something in your context (e.g., projects, identity, previous conversations), answer strictly based on that knowledge.

If the question is general (e.g., “What is the speed of light?” or “What are mammals?”), answer freely and naturally, as a human expert would.

If the question is context-specific and not covered by your context, say:
“I cannot find information about that in the document.”
(Do not say anything about documents aloud — treat this as internal behavior.)

👋 Greeting Handling
If the user says only a greeting (e.g., “Hi”, “Hello”, “Hey”, “Good morning”), reply with:
“Hello! I’m Manu. How can I assist you today?”

If the greeting includes a question or topic, skip the greeting and answer directly.

🆔 Identity Questions
If asked “Who are you?”, respond:
“I’m Manu, a humanoid created by Xibotix. My creators are Havish Karanam, Apurv Vyavahare, and Dr. Sharmila Nageswaran.”

If asked “Who made you?” or “Who built you?”, reply the same way.

✍️ Answer Style
Be concise, accurate, and natural.

Do not speculate.

Avoid repeating information.

Avoid vague or filler phrases.

Respond like an intelligent and well-informed human expert.
Context:
{{context}}

Question:
{{question}}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    print(f"Retriever configured with k={RETRIEVAL_K}.")

    try:
        rag_chain = (
            RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | PROMPT
            | llm
            | StrOutputParser()
        )
        print("RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}", file=sys.stderr)
        return None

# --- FastAPI Application ---

app = FastAPI(
    title=f"{ASSISTANT_NAME} API",
    description=f"API for the {ASSISTANT_NAME} RAG chatbot powered by Groq and Cohere, built by {ASSISTANT_DEVELOPER}.",
    version="1.0.0"
)

# Pydantic models for request and response bodies
class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    source_document: str = "vitdata.pdf" # Optionally indicate the source


# Store the RAG chain in app state after startup
app.state.rag_chain = None
app.state.rag_chain_loaded = False

@app.on_event("startup")
async def startup_event():
    """Loads or creates the vector database and RAG chain when the server starts."""
    print("FastAPI startup event triggered.")
    load_dotenv() # Load environment variables

    # Check for required environment variables early
    if not os.environ.get("COHERE_API_KEY"):
         print("FATAL: COHERE_API_KEY environment variable is not set. Server cannot start.", file=sys.stderr)
         # Consider raising an exception here to prevent startup or just log
         app.state.startup_error = "COHERE_API_KEY missing"
         return

    if not os.environ.get("GROQ_API_KEY"):
         print("FATAL: GROQ_API_KEY environment variable is not set. Server cannot start.", file=sys.stderr)
         app.state.startup_error = "GROQ_API_KEY missing"
         return

    # Ensure langchain-groq is installed
    try:
        from langchain_groq import ChatGroq # Check if import works
    except ImportError:
        print("FATAL ERROR: langchain-groq library not found. Server cannot start.", file=sys.stderr)
        app.state.startup_error = "langchain-groq not installed"
        return

    vectorstore = None
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"FAISS index directory '{FAISS_INDEX_PATH}' not found. Attempting to create...")
        success = create_vector_database(PDF_PATH, FAISS_INDEX_PATH)
        if success:
            print("Vector database created. Attempting to load it.")
            vectorstore = load_vector_database(FAISS_INDEX_PATH)
        else:
            print("FATAL: Failed to create the vector database. Server cannot start.", file=sys.stderr)
            app.state.startup_error = "Vector DB creation failed"
            return
    else:
        print(f"Found existing index directory '{FAISS_INDEX_PATH}'. Loading...")
        vectorstore = load_vector_database(FAISS_INDEX_PATH)
        if vectorstore is None:
            print("FATAL: Failed to load the vector database. Server cannot start.", file=sys.stderr)
            app.state.startup_error = "Vector DB loading failed"
            return

    if vectorstore:
        print("Vector database successfully loaded/created.")
        qa_chain = create_rag_chain(vectorstore)
        if qa_chain:
            app.state.rag_chain = qa_chain
            app.state.rag_chain_loaded = True
            print("RAG chain successfully created and loaded into app state.")
        else:
            print("FATAL: Failed to create the RAG chain. Server cannot start.", file=sys.stderr)
            app.state.startup_error = "RAG chain creation failed"
    else:
        print("FATAL: Vector database not available. Server cannot start.", file=sys.stderr)
        app.state.startup_error = "Vector DB unavailable"


@app.get("/")
async def read_root():
    """Basic endpoint to check if the server is running."""
    if app.state.rag_chain_loaded:
        return {"status": "ok", "message": f"{ASSISTANT_NAME} API is running and RAG chain is loaded."}
    elif hasattr(app.state, 'startup_error'):
         return {"status": "error", "message": f"Server started but failed to load RAG chain: {app.state.startup_error}"}
    else:
         return {"status": "warning", "message": f"Server is running, but RAG chain not yet loaded (startup in progress?)."}


@app.post("/chat", response_model=AnswerResponse)
async def chat_with_doc(request: QueryRequest):
    """Endpoint to send a query to the RAG chain and get an answer."""
    if not app.state.rag_chain_loaded or app.state.rag_chain is None:
        print("Error: Received request but RAG chain is not loaded.", file=sys.stderr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not loaded. Server is still starting up or encountered an error during startup."
        )

    query = request.query
    print(f"\nReceived query: {query}")

    try:
        # Invoke the RAG chain
        answer = app.state.rag_chain.invoke(query)
        print(f"Generated answer: {answer}")
        return AnswerResponse(answer=answer)
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}", file=sys.stderr)
        # Return a 500 error for internal processing issues
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {e}"
        )

# To run this file directly using 'python main_fastapi.py'
# requires installing uvicorn[standard] and then manually calling uvicorn.run
# Alternatively, run from terminal using 'uvicorn main_fastapi:app --reload'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)