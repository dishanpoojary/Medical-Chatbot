from flask import Flask, render_template, jsonify, request, session
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
import uuid
import json
from datetime import datetime
from flask_session import Session
import redis

app = Flask(__name__)
CORS(app, supports_credentials=True)

# --- Load Environment Variables ---
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

# --- Session Configuration ---
app.secret_key = os.environ.get('SESSION_SECRET', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'  # Can use 'redis' for production
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
Session(app)

# Initialize conversation history storage
conversation_history = {}

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
        
        try:
            llm = ChatOpenAI(
                model=nvidia_model_name,
                openai_api_key=NVIDIA_API_KEY,
                openai_api_base=nvidia_base_url,
                temperature=0.7,
                max_tokens=1000
            )
            print(f"✅ LLM initialized ({nvidia_model_name}).")
        except Exception as e:
            print(f"⚠️ NVIDIA endpoint failed: {e}")
            print("⚠️ Using a fallback model...")
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

# --- Helper Functions for Conversation Management ---
def get_user_id():
    """Get or create user session ID"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']

def get_current_chat_id():
    """Get current active chat ID"""
    user_id = get_user_id()
    if 'current_chat_id' not in session:
        # Create new chat
        chat_id = str(uuid.uuid4())
        session['current_chat_id'] = chat_id
        initialize_chat_history(user_id, chat_id)
    return session['current_chat_id']

def initialize_chat_history(user_id, chat_id):
    """Initialize a new chat history"""
    if user_id not in conversation_history:
        conversation_history[user_id] = {}
    
    conversation_history[user_id][chat_id] = {
        'id': chat_id,
        'title': 'New Chat',
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'messages': []
    }

def add_message_to_chat(user_id, chat_id, role, content):
    """Add a message to chat history"""
    if user_id in conversation_history and chat_id in conversation_history[user_id]:
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        conversation_history[user_id][chat_id]['messages'].append(message)
        conversation_history[user_id][chat_id]['updated_at'] = datetime.now().isoformat()
        
        # Update chat title if this is the first user message
        if len(conversation_history[user_id][chat_id]['messages']) == 1:
            title = content[:50] + "..." if len(content) > 50 else content
            conversation_history[user_id][chat_id]['title'] = title

def get_chat_history(user_id, chat_id):
    """Get chat history for a specific chat"""
    if user_id in conversation_history and chat_id in conversation_history[user_id]:
        return conversation_history[user_id][chat_id]
    return None

def get_user_chats(user_id):
    """Get all chats for a user"""
    if user_id in conversation_history:
        chats = list(conversation_history[user_id].values())
        # Sort by updated time (newest first)
        chats.sort(key=lambda x: x['updated_at'], reverse=True)
        return chats
    return []

def create_new_chat(user_id):
    """Create a new chat and return its ID"""
    chat_id = str(uuid.uuid4())
    initialize_chat_history(user_id, chat_id)
    session['current_chat_id'] = chat_id
    return chat_id

def switch_chat(user_id, chat_id):
    """Switch to a different chat"""
    if user_id in conversation_history and chat_id in conversation_history[user_id]:
        session['current_chat_id'] = chat_id
        return True
    return False

# Initialize on startup
try:
    initialize_components()
except Exception as e:
    print(f"⚠️ Initialization failed: {e}")

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/api/init", methods=["GET"])
def initialize_session():
    """Initialize user session and return initial data"""
    user_id = get_user_id()
    chat_id = get_current_chat_id()
    
    chats = get_user_chats(user_id)
    current_chat = get_chat_history(user_id, chat_id)
    
    return jsonify({
        "user_id": user_id,
        "current_chat_id": chat_id,
        "chats": chats,
        "current_chat": current_chat,
        "error": False
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat requests with conversation history"""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"response": "Please enter a message.", "error": False})
        
        # Get user and chat info
        user_id = get_user_id()
        chat_id = get_current_chat_id()
        
        print(f"Processing query from user {user_id} in chat {chat_id}: {message}")
        
        # Initialize components if not already done
        if rag_chain is None:
            initialize_components()
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": message})
        answer = response.get("answer", "Sorry, I couldn't find an answer to your question.")
        
        # Format the response
        formatted_answer = format_response(answer)
        
        # Save messages to history
        add_message_to_chat(user_id, chat_id, "user", message)
        add_message_to_chat(user_id, chat_id, "assistant", formatted_answer)
        
        print(f"Response generated successfully")
        return jsonify({
            "response": formatted_answer,
            "error": False,
            "chat_id": chat_id
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({
            "response": "Sorry, I'm experiencing technical difficulties. Please try again later.",
            "error": True
        })

@app.route("/api/chats", methods=["GET"])
def get_chats():
    """Get all chats for current user"""
    user_id = get_user_id()
    chats = get_user_chats(user_id)
    
    return jsonify({
        "chats": chats,
        "error": False
    })

@app.route("/api/chats/new", methods=["POST"])
def new_chat():
    """Create a new chat"""
    user_id = get_user_id()
    chat_id = create_new_chat(user_id)
    
    return jsonify({
        "chat_id": chat_id,
        "message": "New chat created",
        "error": False
    })

@app.route("/api/chats/<chat_id>/switch", methods=["POST"])
def switch_to_chat(chat_id):
    """Switch to a specific chat"""
    user_id = get_user_id()
    
    if switch_chat(user_id, chat_id):
        chat_history = get_chat_history(user_id, chat_id)
        return jsonify({
            "message": "Switched to chat",
            "chat_history": chat_history,
            "error": False
        })
    else:
        return jsonify({
            "message": "Chat not found",
            "error": True
        }), 404

@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    """Get specific chat history"""
    user_id = get_user_id()
    chat_history = get_chat_history(user_id, chat_id)
    
    if chat_history:
        return jsonify({
            "chat": chat_history,
            "error": False
        })
    else:
        return jsonify({
            "message": "Chat not found",
            "error": True
        }), 404

@app.route("/api/chats/<chat_id>/delete", methods=["DELETE"])
def delete_chat(chat_id):
    """Delete a chat"""
    user_id = get_user_id()
    
    if user_id in conversation_history and chat_id in conversation_history[user_id]:
        del conversation_history[user_id][chat_id]
        
        # If this was the current chat, create a new one
        if session.get('current_chat_id') == chat_id:
            create_new_chat(user_id)
        
        return jsonify({
            "message": "Chat deleted",
            "error": False
        })
    else:
        return jsonify({
            "message": "Chat not found",
            "error": True
        }), 404

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag_chain is not None,
        "session_enabled": True
    })

def format_response(text):
    """Format the response for better readability while preserving formatting"""
    # Remove excessive whitespace but preserve intentional line breaks
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean up each line but keep it as a separate line
        line = ' '.join(line.split())  # Clean within the line
        if line.strip():  # Only add non-empty lines
            cleaned_lines.append(line)
    
    # Join back with proper spacing
    formatted_text = '\n\n'.join(cleaned_lines)
    
    # Ensure proper punctuation at the end
    if formatted_text and not formatted_text.endswith(('.', '!', '?', ':')):
        formatted_text += '.'
    
    return formatted_text

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)