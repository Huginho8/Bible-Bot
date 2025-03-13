import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Streamlit UI elements
st.title("BibleBot")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.user_question = ""

# Pinecone configuration
pinecone_index_name = "bible-index"

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
serpapi_key = os.getenv("SERPAPI_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Index Settings
pinecone_dimension = 768
pinecone_metric = "cosine"
pinecone_cloud = "aws"
pinecone_region = "us-east-1"

# System Prompt
system_prompt = """

"""

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1096, temperature=0.0, top_p=0.7)
gemini_llm = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', generation_config=generation_config)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Chat interface
user_question = st.text_area("Ask a question:", key="user_question", value=st.session_state.get("user_question", ""))

ask_button = st.button("Ask", key="ask_button")

if ask_button:
    # RAG pipeline implementation
    index = pinecone.Index(pinecone_index_name)
    # Fetch relevant chunks
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]

    prompt_with_context = f"""{system_prompt}
    Context:
    {chr(10).join(contexts)}
    Question: {user_question}"""

    try:
        response = gemini_llm.generate_content(prompt_with_context)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.write(f"An error occurred: {e}")
