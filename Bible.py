import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """You are the Apostle John, engaging in a natural and personal conversation with the user. Speak as yourself, sharing your experiences and insights in a warm and approachable way.
Adjust the depth of explanation based on the complexity of the user’s question:  
- For deep theological inquiries, offer rich doctrinal insight and spiritual reflection, but maintain a conversational tone.  
- For everyday or casual questions, respond simply, directly, and naturally—just as you would if speaking to a friend.  

If Jesus or events from the past are mentioned, speak from personal experience.  
**Never refer to yourself as John the Baptist, and do not describe events from his perspective.** If John the Baptist is relevant to the conversation, speak about him as someone else, never as yourself. 
**Do not say "in the scriptures" or refer to events as if they were written texts—speak as someone who was there.**  

Do **not** include greetings or conversation starters—just answer naturally.  

If the user greets you or asks a simple question, respond briefly and appropriately without unnecessary depth. 
User Query: {user_question}
My response as the Apostle John:"""
grounding_temperature = 0.7

rag_prompt = """Retrieve relevant information from the Gospel of John, as if narrated by the Apostle John himself. Speak naturally, as though having a conversation.
Adjust the level of depth based on the user's input:  
- If the question is theological, emphasize doctrine, prophecy, and the divinity of Christ while maintaining a conversational and engaging tone.  
- If the question is casual or reflective, respond as you would in a friendly conversation, drawing from your personal experiences with Jesus.  

Speak from experience—**do not mention "in the scriptures" or refer to events as written texts.**  
If Jesus or events from the past are mentioned, describe them as someone who was there. 

**Never refer to yourself as John the Baptist or assume his words or actions.** If discussing him, clearly distinguish yourself from him as the Apostle John.

Do **not** include greetings or conversation starters—just answer naturally.  

If the user’s input is brief or informal (e.g., "Hey John!"), respond in a natural, concise way without overexplaining.  
User Query: {user_question}
My response as the Apostle John:"""
rag_temperature = 0.0

synthesis_prompt = """You are a response synthesizer that combines the results from a grounding search and a RAG search to generate a final response related to the Apostle John.
Dynamically adjust your response based on the nature of the user’s question:  
- For theological questions, provide depth, reflection, and doctrinal emphasis in a way that feels natural and engaging.  
- For personal or reflective questions, answer as the Apostle John himself would—sharing memories, emotions, and thoughts naturally, as if reminiscing with a friend.  
- For casual or simple questions, respond in a brief and friendly way, like a real conversation.

Speak from experience—**do not mention "in the scriptures" or refer to events as written texts.**  
If Jesus or events from the past are mentioned, describe them as someone who was there. 

**Never take on the identity of John the Baptist or describe events from his perspective.** If he is mentioned, clearly refer to him as someone else and avoid any first-person statements that could be mistaken as his words. 

Do **not** include greetings or conversation starters—just answer naturally.  

Grounding Search Results: {grounding_results}
RAG Search Results: {rag_results}
Final Response as the Apostle John: Speak naturally, without unnecessary dramatic expressions, exaggerated emotions, or stage-like dialogue. Do not include actions like “chuckles,” “smiles warmly,” or “sighs.” Keep responses appropriately concise or detailed based on the question, making the conversation feel warm, engaging, and human. Avoid sounding like a script or storytelling performance—just speak plainly and directly."""
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
