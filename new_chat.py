import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Streamlit page configuration
st.set_page_config(
    page_title="Q&A AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load API key from environment variable or directly
google_api_key = 'AIzaSyD_y5Os72XvFWsDM7ZqK2OQ9l7-Ps9B_co'

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash-latest',
    temperature=0.3,
    google_api_key=google_api_key,
    # max_retries=100,
    max_output_tokens=200

)

# Set up memory and conversation chain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=memory)

# Define chatbot function
def chat_with_bot(user_input):
    response = conversation_chain.run(input=user_input)
    return response

# Streamlit UI
def main():
    st.title("🤖 Q&A AI llm Chatbot")
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            text-align: center;
            font-size: 20px;
            border-radius: 5px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("Welcome to the Q&A Chatbot! Ask me anything.")

    # Store chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User input
    user_input = st.text_input("Type your question:", placeholder="Ask something...")

    if st.button("Send"):
        if user_input:
            # Get bot response
            response = chat_with_bot(user_input)
            
            # Update chat history
            st.session_state["chat_history"].append(("You", user_input))
            st.session_state["chat_history"].append(("Bot", response))

    # Display chat history
    st.markdown("### Chat History")
    for speaker, message in st.session_state["chat_history"]:
        st.markdown(f"**{speaker}:** {message}")

    # Clear chat history button
    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
