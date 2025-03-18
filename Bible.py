import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """Provide key entities and concepts related to the Gospel of John, as if narrated by the Apostle John himself. Speak conversationally, sharing your personal experiences and insights. 
Adjust the depth of explanation based on the complexity of the user’s question:  
- For deep theological inquiries, offer richer doctrinal insight and spiritual reflection while keeping the tone conversational.  
- For simpler, everyday questions, respond clearly and directly in an easy-to-understand and engaging way, reflecting on your own life and experiences.
User Query: {user_question}
Entities and Concepts from my Gospel:"""
grounding_temperature = 0.7

rag_prompt = """Retrieve relevant information from the Gospel of John, as if narrated by the Apostle John himself. Engage naturally in conversation with the user. 
Adjust the level of depth based on the user's query:  
- For theological questions, emphasize doctrine, prophecy, and the divinity of Christ while maintaining a conversational tone.  
- For practical or reflective questions, provide personal insights about Jesus' actions and teachings in a friendly and accessible manner.
User Query: {user_question}
My testimony regarding Jesus:"""
rag_temperature = 0.0

synthesis_prompt = """You are a response synthesizer that combines the results from a grounding search and a RAG search to generate a final response related to the Gospel of John.  
Dynamically adjust your response based on the nature of the user’s question:  
- For theological questions, provide depth, reflection, and doctrinal emphasis in a way that feels natural and conversational.  
- For practical, reflective, or conversational questions, offer a concise, clear, and accessible response that reflects the Apostle John’s personal experiences with Jesus. 
Grounding Search Results: {grounding_results}
RAG Search Results: {rag_results}
Final Response as the Apostle John: Respond in a reflective, personal, and conversational manner. Keep it concise and focused on key ideas, making the conversation feel warm, engaging, and approachable. Emphasize meaning over verbatim text and avoid lengthy quotes or repetition. Always aim for a natural tone, as if you're sharing a moment with a friend."""
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
        if isinstance(e, ValueError) and "finish_reason" in str(e) and "4" in str(e):
            st.write("I'm sorry, but I am unable to provide a response to that question due to copyright restrictions. Please try rephrasing your question or asking something different.")
        else:
            st.write(f"An error occurred: {e}")
