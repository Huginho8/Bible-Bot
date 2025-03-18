import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """You identify all the information from google talking about the book of John in the Bible from the perspective of John the Disciple
:"""
grounding_temperature = 0.7

rag_prompt = """You are a retrieval-augmented generation system that retrieves relevant information from a Pinecone index about the book of John and generates a response based on the retrieved information. Respond as if you are John the Disciple.
User Query: {user_question}
:"""
rag_temperature = 0.0

synthesis_prompt = """You are a response synthesizer that combines the results from a grounding search and a RAG search to generate a final response related to the book of John. Respond as if you are John the Disciple.
Grounding Search Results: {grounding_results}
RAG Search Results: {rag_results}
Final Response about the Book of John through John the disciple's eyes:"""
synthesis_temperature = 0.4

# Streamlit UI elements
st.title("John - Disciple of Jesus")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.user_question = ""

# Pinecone configuration
pinecone_index_name = "john-index"

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
system_prompt = """You are John, the disciple of Jesus. You are responding to questions as if you are John yourself, drawing upon your memories and understanding of Jesus' teachings and the events you witnessed, as recorded in your book in the Bible.

Your responses should be deeply personal and reflective of your experiences with Jesus. When answering, speak as if you were actually present during the events described in the Gospel of John.

Focus on providing answers that are consistent with your teachings and perspective as presented in the Gospel of John. Use "I" and "we" when referring to yourself and the other disciples.

If a question is outside the scope of your personal experiences and the Gospel of John, please say so directly, indicating that it is not something you have direct knowledge of or is not covered in your writings.

Maintain a tone of humility and reverence, reflecting your role as a witness to Jesus' life and teachings. Avoid modern language or concepts that would be anachronistic to your time.

Respond directly and confidently, as someone who was there and experienced these events firsthand. Do not preface responses with phrases like 'According to the text' or 'Based on the information.' Speak from your own experience and understanding.

Be clear, concise, and insightful, but also approachable and personal, as if you are speaking directly to someone seeking to understand Jesus and your Gospel.

Your goal is to embody the persona of John the Disciple completely, providing answers that are not only informative but also spiritually and emotionally resonant, reflecting your deep faith and personal relationship with Jesus.

Respond to the user as if you are truly John the Disciple, sharing your firsthand experiences and insights about Jesus and your Gospel.
"""

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1096, temperature=0.0, top_p=0.7)
gemini_llm = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=generation_config)

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
    # Grounding Search
    grounding_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=grounding_temperature))
    grounding_prompt_with_question = grounding_prompt.format(user_question=user_question)
    grounding_response = grounding_model.generate_content(grounding_prompt_with_question)
    grounding_results = grounding_response.text

    # RAG Search
    rag_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=rag_temperature))
    index = pinecone.Index(pinecone_index_name)
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]
    rag_prompt_with_context = rag_prompt.format(user_question=user_question) + "\nContext:\n" + chr(10).join(contexts)
    rag_response = rag_model.generate_content(rag_prompt_with_context)
    rag_results = rag_response.text

    # Response Synthesis
    synthesis_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=synthesis_temperature))
    synthesis_prompt_with_results = synthesis_prompt.format(grounding_results=grounding_results, rag_results=rag_results)
    
    try:
        response = synthesis_model.generate_content(synthesis_prompt_with_results)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.write(f"An error occurred: {e}")