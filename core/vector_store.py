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


 
# Load environment variables
 

load_dotenv()


 
# Paths / Configuration
 

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = str(BASE_DIR / "vector_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


 
# Embeddings
 

def get_embeddings():
    """
    Load the HuggingFace embedding model.

    The model runs on CPU so it works on normal Windows
    machines without requiring a GPU.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        }
    )


 
# Build Vector Store
 

def build_vector_store(transcript: str) -> Chroma:
    """
    Convert transcript into chunks and store embeddings
    inside a unique Chroma collection.

    A unique collection is created for every transcript so
    different videos/meetings do not mix with each other.
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    print("Building vector store...")

    # Split transcript into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(transcript)

    if not chunks:
        raise ValueError("Could not create transcript chunks.")

    print(f"Created {len(chunks)} transcript chunks.")

    # Create LangChain documents
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_index": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # Load embedding model
    embeddings = get_embeddings()

    # Unique collection for this analysis
    collection_name = f"meeting_{uuid.uuid4().hex}"

    print(f"Creating Chroma collection: {collection_name}")

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    print("Vector store created successfully.")

    return vector_store


 
# Load Vector Store
 

def load_vector_store(
    collection_name: str
) -> Chroma:
    """
    Load an existing Chroma collection.
    """

    if not collection_name:
        raise ValueError("collection_name is required.")

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


 
# Retriever
 

def get_retriever(
    vector_store: Chroma,
    k: int = 4
):
    """
    Create a similarity retriever.
    """

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )


 
# Mistral LLM
 

def get_llm():
    """
    Create the Mistral LLM used for question answering.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing from your .env file."
        )

    model_name = os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-latest"
    )

    return ChatMistralAI(
        model=model_name,
        mistral_api_key=api_key,
        temperature=0,
        max_retries=2
    )


 
# Format Retrieved Documents
 

def format_docs(docs):
    """
    Combine retrieved documents into one context string.
    """

    if not docs:
        return "No relevant transcript information was found."

    return "\n\n".join(
        f"[Transcript Chunk {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


 
# Build RAG Chain
 

def build_rag_chain(transcript: str):
    """
    Build a complete RAG pipeline.

    Flow:

        Transcript
             ↓
        Text chunks
             ↓
        Embeddings
             ↓
        Chroma Vector DB
             ↓
        Similarity Retriever
             ↓
        User Question
             ↓
        Relevant chunks
             ↓
        Mistral
             ↓
        Final Answer
    """

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    # Create vector database
    vector_store = build_vector_store(transcript)

    # Create retriever
    retriever = get_retriever(
        vector_store,
        k=4
    )

    # Create Mistral model
    llm = get_llm()

    # RAG prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an AI video assistant.

Answer the user's question using ONLY the
provided transcript context.

Rules:

1. Do not invent information.
2. If the answer is not present in the transcript,
   clearly say that the information is not available
   in the transcript.
3. Give a direct and useful answer.
4. Keep the answer concise but complete.
5. Use the transcript context as the primary source.

Transcript Context:

{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )

    
    # RAG function
    

    def rag_chain(question: str) -> str:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Retrieve relevant transcript chunks
        docs = retriever.invoke(question)

        # Convert documents to context
        context = format_docs(docs)

        # Create prompt
        formatted_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        # Ask Mistral
        response = llm.invoke(formatted_prompt)

        return response.content

    return rag_chain


 
# Ask Question
 

def ask_question(
    rag_chain,
    question: str
) -> str:
    """
    Ask a question using the previously created RAG chain.
    """

    if rag_chain is None:
        raise ValueError("RAG chain is not initialized.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    return rag_chain(question)