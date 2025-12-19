from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
from pinecone import Pinecone
import os
import time

app = Flask(__name__)
CORS(app)

# --- Load Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

# --- Initialize Components ---
embeddings = None
vectorstore = None
retriever = None
llm = None
rag_chain = None

def initialize_components():
    """Initialize all RAG components"""
    global embeddings, vectorstore, retriever, llm, rag_chain
    
    try:
        # Initialize Embeddings
        print("Initializing embeddings...")
        embeddings = download_hugging_face_embeddings()
        print("✅ Embeddings loaded.")
        
        # Initialize Pinecone
        print("Initializing Pinecone...")
        index_name = "medical-chatbot"
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        print("✅ Pinecone vector store connected.")
        
        # Initialize Retriever
        print("Initializing Retriever...")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("✅ Retriever created.")
        
        # Initialize LLM
        print("Initializing LLM via NVIDIA endpoint...")
        nvidia_base_url = "https://integrate.api.nvidia.com/v1"
        nvidia_model_name = "openai/gpt-oss-120b"
        
        # You can also use a fallback model if NVIDIA is not available
        try:
            llm = ChatOpenAI(
                model=nvidia_model_name,
                openai_api_key=NVIDIA_API_KEY,
                openai_api_base=nvidia_base_url,
                temperature=0.7,
                max_tokens=500
            )
            print(f"✅ LLM initialized ({nvidia_model_name}).")
        except Exception as e:
            print(f"⚠️ NVIDIA endpoint failed: {e}")
            print("⚠️ Using a fallback model...")
            # Fallback to a local model or different API
            # You might want to use Ollama or another provider here
            raise e
        
        # Define Prompt and Chain
        print("Defining prompt and chain...")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("✅ RAG chain created.")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        # You might want to set up a degraded mode or show a maintenance page

# Initialize on startup
try:
    initialize_components()
except Exception as e:
    print(f"⚠️ Initialization failed: {e}")

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"response": "Please enter a message.", "error": False})
        
        print(f"Processing query: {message}")
        
        # Initialize components if not already done
        if rag_chain is None:
            initialize_components()
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": message})
        answer = response.get("answer", "Sorry, I couldn't find an answer to your question.")
        
        # Format the response
        formatted_answer = format_response(answer)
        
        print(f"Response generated successfully")
        return jsonify({
            "response": formatted_answer,
            "error": False
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({
            "response": "Sorry, I'm experiencing technical difficulties. Please try again later.",
            "error": True
        })

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag_chain is not None
    })

def format_response(text):
    """Format the response for better readability"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Ensure the response ends with a proper punctuation
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)