import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="DocBot", page_icon="🤖", layout="centered")
st.title("Document QA Bot 🤖")
st.write("Upload a PDF and ask questions! Your document is processed securely and is not stored after you leave.")

# --- API KEY HANDLING ---
# (Optional) Hardcode your keys here if you don't want to use the sidebar
# NOTE: This is NOT recommended for public apps.
# COHERE_API_KEY = "YOUR_API_KEY_HERE"
# PINECONE_API_KEY = "YOUR_API_KEY_HERE"

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys 🔑")
    try:
        # Try to use hardcoded keys if they exist
        cohere_api_key = C0OhThNGBaMlvsFMcbI2rqnmimpDFgHh8CYBeCXN
        st.success("Cohere key loaded from code.", icon="✔")
    except NameError:
        # Fallback to text input
        cohere_api_key = st.text_input("Cohere API Key", type="password")

    try:
        # Try to use hardcoded keys if they exist
        pinecone_api_key = pcsk_4ADH46_RZawnoPXbJ7D4eCVDgL8qvYqGvocYe1xx7zVN938RtQB654w7B9aJKZUmqspxtV
        st.success("Pinecone key loaded from code.", icon="✔")
    except NameError:
        # Fallback to text input
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")

# --- SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

# --- FILE UPLOAD & PROCESSING ---
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

if uploaded_file:
    # Check if this is a new file. If so, process it.
    if uploaded_file.file_id != st.session_state.processed_file_id:
        if not cohere_api_key or not pinecone_api_key:
            st.error("Please provide your Cohere and Pinecone API keys in the sidebar.")
        else:
            with st.spinner("Processing document... This may take a moment."):
                try:
                    # Save uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Initialize VectorStore and Chatbot
                    st.session_state.vector_store = VectorStore(cohere_api_key, pinecone_api_key)
                    
                    # Process the document
                    st.session_state.vector_store.process_document(tmp_file_path)
                    
                    # Initialize Chatbot
                    st.session_state.chatbot = Chatbot(st.session_state.vector_store, cohere_api_key)
                    
                    # Store the ID of the processed file
                    st.session_state.processed_file_id = uploaded_file.file_id
                    
                    # Clear chat history for the new document
                    st.session_state.chat_history = []
                    
                    st.success("Document processed successfully! You can now ask questions.")
                    
                    # Clean up the temporary file
                    os.unlink(tmp_file_path)
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    # Clean up the temp file even if an error occurs
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)

# --- CHAT INTERFACE ---

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if document is processed
    if not st.session_state.chatbot:
        st.error("Please upload a document before asking questions.")
    else:
        # 1. Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate and display bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Get the stream from the chatbot
                response_stream = st.session_state.chatbot.respond(prompt)
                
                # Iterate over the stream and display "typing" effect
                for event in response_stream:
                    if event.event_type == "text-generation":
                        full_response += event.text
                        message_placeholder.markdown(full_response + "▌") # Blinking cursor
                
                # Display the final, complete response
                message_placeholder.markdown(full_response)
                
                # 3. Add final bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"An error occurred while generating response: {e}")
