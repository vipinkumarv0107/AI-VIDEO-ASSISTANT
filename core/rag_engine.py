import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = str(BASE_DIR / "vector_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-small-latest"
)


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings():
    print("Loading embedding model...")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
    )


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vector_store(transcript: str) -> Chroma:

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_text(transcript)

    if not chunks:
        raise RuntimeError(
            "No transcript chunks were created."
        )

    print(f"Created {len(chunks)} transcript chunks.")

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # Unique collection for every analysis
    collection_name = (
        f"meeting_{uuid.uuid4().hex}"
    )

    print(
        f"Creating Chroma collection: {collection_name}"
    )

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    print("Vector store created successfully.")

    return vector_store


# ============================================================
# RETRIEVER
# ============================================================

def get_retriever(
    vector_store: Chroma,
    k: int = 4
):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        },
    )


# ============================================================
# MISTRAL LLM
# ============================================================

def get_llm():

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing from your .env file."
        )

    print(
        f"Using Mistral model: {MISTRAL_MODEL}"
    )

    return ChatMistralAI(
    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    mistral_api_key=api_key,
    temperature=0,
    max_retries=0,
)

# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    if not docs:
        return (
            "No relevant transcript information "
            "was found."
        )

    return "\n\n".join(
        f"[Transcript Chunk {i + 1}]\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_rag_chain(
    transcript: str
):

    if not transcript or not transcript.strip():
        raise ValueError(
            "Transcript is empty."
        )

    print("Building RAG chain...")

    # Create vector database
    vector_store = build_vector_store(
        transcript
    )

    # Retriever
    retriever = get_retriever(
        vector_store,
        k=4
    )

    # Mistral
    # llm = get_llm()
    analysis = analyze_transcript(transcript)

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI video assistant.

Answer the user's question using ONLY
the provided transcript context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the
   transcript, say:

"I couldn't find that information in
the transcript."

4. Give a direct and useful answer.
5. Keep the answer concise but complete.

Transcript Context:

{context}
""",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    # ========================================================
    # RAG FUNCTION
    # ========================================================

    def rag_chain(
        question: str
    ):

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        print(
            f"RAG question: {question}"
        )

        # Retrieve relevant transcript chunks
        docs = retriever.invoke(
            question
        )

        print(
            f"Retrieved {len(docs)} transcript chunks."
        )

        # Format context
        context = format_docs(
            docs
        )

        # Build prompt
        formatted_prompt = prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        # ONE MISTRAL REQUEST
        response = llm.invoke(
            formatted_prompt
        )

        return response.content

    return rag_chain


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    rag_chain,
    question: str
):

    if not question or not question.strip():
        return "Please enter a question."

    if rag_chain is None:
        raise RuntimeError(
            "RAG chain is not initialized."
        )

    return rag_chain(
        question.strip()
    )