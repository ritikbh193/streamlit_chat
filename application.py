
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import DocArrayInMemorySearch

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder

# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key='AIzaSyAJLv_QjBn1QPliUJ6_CTR4peHzd2cXVYg')

# import nltk
# nltk.download('punkt')


# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter

# # Load the PDF file
# pdf_path = "/content/Updated Remark App Description.pdf"
# loader = PyPDFLoader(pdf_path)

# # Load the content of the PDF
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(documents)

# # Display the result
# for i, split in enumerate(splits):
#     print(f"Split {i + 1}:")
#     print(split)
#     print("\n" + "="*80 + "\n")


# from langchain_text_splitters import NLTKTextSplitter

# text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)

# chunks = text_splitter.split_documents(documents=documents)

# print(len(chunks))

# print(type(chunks[0]))


# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAJLv_QjBn1QPliUJ6_CTR4peHzd2cXVYg")

# # Store the chunks in vector store
# from langchain_community.vectorstores import Chroma

# # Embed each chunk and load it into the vector store
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")

# # Persist the database on drive
# db.persist()


# db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)


# # Converting CHROMA db_connection to Retriever Object
# retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# print(type(retriever))


# chat_template = ChatPromptTemplate.from_messages([
#     # System Message Prompt Template
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark.
#                   Given a context and question from user,
#                   you should answer based on the given context and ask if needed the more information."""),
#     # Human Message Prompt Template
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information, you are a personal assistant for user of Remark Job And Recruiter Portal.
#     Context: {context}

#     Modify the answer in your way & if you can't answer it from our context then please generate the answer your own way but make sure that you are a AI bot from Remark Job And Recruiter Portal so make sure that the answer will be relevant to Remark Job And Recruiter Portal if the question is not relevant to the job and recruitment industry, then do make any answer.
#     Please don't tell the users that you have got any context from remark, also remember the details user shared for particular session, so that you can answer them properly.

#     Question: {question}
#     Answer: """)
# ])

# document_chain = create_stuff_documents_chain(model, chat_template)


# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()


# from langchain_core.runnables import RunnablePassthrough

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# import sys
# from IPython.display import Markdown as md

# all_chunks = []

# def chat_with_remark(user_input):
#     for chunk in rag_chain.stream("how can i connect you with flutter app"):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

#     return all_chunks


# from IPython.display import Markdown as md

# user_input = "";

# def ask_user():
#     user_input = input("Ask Question")
#     print(user_input)
#     generate_answer(user_input)
#     print("--------------------------------------------------------------------------")
#     ask_user()
#     sys.stdout.flush()


# def generate_answer(get_user_input):
#     for chunk in rag_chain.stream(get_user_input):
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

# ask_user()








# '''code 2'''

# # Download NLTK punkt if needed
# import nltk
# nltk.download('punkt')








# import streamlit as st
# import sys
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.vectorstores import DocArrayInMemorySearch

# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.memory import ChatMessageHistory
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import MessagesPlaceholder

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter

# # Download NLTK punkt if needed
# import nltk
# nltk.download('punkt_tab')


# # Initialize the model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key='AIzaSyC6ckk_vsOEl74yQpfq1UYxr5xTaV-jITg')

# # PDF loading and splitting
# pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # Embed the chunks and store them in a vector store
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyC6ckk_vsOEl74yQpfq1UYxr5xTaV-jITg')
# from langchain_community.vectorstores import Chroma

# # Create Chroma vectorstore and persist it
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
# db.persist()

# # Connect to the Chroma vectorstore
# db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# # Create the retriever
# retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# # Create chat template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""),
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information. You are a personal assistant for the Remark Job And Recruiter Portal. 
#     Context: {context}

#     Question: {question}
#     Answer: """)
# ])

# # Create document chain and output parser
# document_chain = create_stuff_documents_chain(model, chat_template)
# output_parser = StrOutputParser()

# # Chain for retrieval and answering
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# # Streamlit app setup
# st.title("ChatBot")

# # Initialize session state for storing chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Function to handle chat and store response
# def chat_with_remark(user_input):
#     all_chunks = []
#     for chunk in rag_chain.stream(user_input):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()
    
#     return ''.join(all_chunks)

# # Input for user question
# user_input = st.text_input("Ask a Question:", key="user_input")

# # If there's user input, generate a response
# if user_input:
#     answer = chat_with_remark(user_input)
#     st.session_state.chat_history.append({"question": user_input, "answer": answer})

# # Display chat history
# if st.session_state.chat_history:
#     st.write("### Chat ")
#     for entry in st.session_state.chat_history:
#         st.write(f"{entry['question']}")
#         st.write(f"{entry['answer']}")
#         st.write("---")

# # Button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []




# '''code 3'''


# import streamlit as st
# import sys
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain.vectorstores import DocArrayInMemorySearch

# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.memory import ChatMessageHistory
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts import ChatPromptTemplate
# from langchain.prompts.chat import MessagesPlaceholder

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter
# # Additional imports for Chroma deletion
# from langchain_community.vectorstores import Chroma

# # Download NLTK punkt if needed
# import nltk
# nltk.download('punkt_tab')


# # Initialize the model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key='AIzaSyC6ckk_vsOEl74yQpfq1UYxr5xTaV-jITg')

# # PDF loading and splitting
# pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # Embed the chunks and store them in a vector store
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyC6ckk_vsOEl74yQpfq1UYxr5xTaV-jITg')
# from langchain_community.vectorstores import Chroma

# # Create Chroma vectorstore and persist it
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
# db.persist()

# # Connect to the Chroma vectorstore
# db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# # Create the retriever
# retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# # Create chat template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""),
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information. You are a personal assistant for the Remark Job And Recruiter Portal. 
#     Context: {context}

#     Question: {question}
#     Answer: """)
# ])

# # Create document chain and output parser
# document_chain = create_stuff_documents_chain(model, chat_template)
# output_parser = StrOutputParser()

# # Chain for retrieval and answering
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# # Streamlit app setup
# st.title("ChatBot")

# # Initialize session state for storing chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []


# # Function to delete the entire Chroma collection
# def clear_chroma_db():
#     # Load the vectorstore
#     db = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

#     # Delete the entire collection
#     db.delete_collection()
#     st.success("Chroma DB collection has been deleted.")


# # Function to handle chat and store response
# def chat_with_remark(user_input):
#     # Check if user input is 'clear' and clear the Chroma database
#     if user_input.lower() == "clear":
#         clear_chroma_db()  # Clear Chroma DB
#         return "Chroma DB and chat history have been cleared."

#     all_chunks = []
#     for chunk in rag_chain.stream(user_input):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

#     return ''.join(all_chunks)

# # Input for user question
# user_input = st.text_input("Ask a Question:", key="user_input")

# # If there's user input, generate a response
# if user_input:
#     answer = chat_with_remark(user_input)
#     st.session_state.chat_history.append({"question": user_input, "answer": answer})

# # Display chat history
# if st.session_state.chat_history:
#     st.write("### Chat ")
#     for entry in st.session_state.chat_history:
#         st.write(f"{entry['question']}")
#         st.write(f"{entry['answer']}")
#         st.write("---")

# # Button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     clear_chroma_db()  # Also clear the Chroma DB when the button is clicked











'''code 4'''
import streamlit as st
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
# from langchain.vectorstores import DocArrayInMemorySearch

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.prompts.chat import MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma  # Only import once

# Saving conversation history with this code
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# Download NLTK punkt if needed
import nltk
nltk.download('punkt')  # Correct the download name



import os
# Load Google API Key from environment variable for security
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=google_api_key)


# Set up memory and conversation chain
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=model, memory=memory)


# PDF loading and splitting
pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the content using NLTKTextSplitter
text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Embed the chunks and store them in a vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Create Chroma vectorstore and persist it
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Connect to the Chroma vectorstore
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Create the retriever
retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# Create chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information also save user details like name description and more. You are a personal assistant for the Remark Job And Recruiter Portal. 
    Context: {context}

    Question: {question}
    Answer: """)
])

# Create document chain and output parser
document_chain = create_stuff_documents_chain(model, chat_template)
output_parser = StrOutputParser()

# Chain for retrieval and answering
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | model
    | output_parser
)

# Streamlit app setup
st.title("ChatBot")

# Initialize session state for storing chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to delete the entire Chroma collection
def clear_chroma_db():
    # Load the vectorstore
    db = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
    # Delete the entire collection
    db.delete_collection()
    st.success("Deleted ")

# Function to handle chat and store response
def chat_with_remark(user_input):
    # Check if user input is 'clear' and clear the Chroma database
    if user_input.lower() == "clear":
        clear_chroma_db()  # Clear Chroma DB
        return "Chroma DB and chat history have been cleared."

    all_chunks = []
    for chunk in rag_chain.stream(user_input):
        all_chunks.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()

    return ''.join(all_chunks)

# Input for user question
user_input = st.text_input("Ask a Question:", key="user_input")

# If there's user input, generate a response
if user_input:
    answer = chat_with_remark(user_input)
    st.session_state.chat_history.append({"question": user_input, "answer": answer})

# Display chat history
if st.session_state.chat_history:
    st.write("### Chat")
    for entry in st.session_state.chat_history:
        st.write(f" {entry['question']}")
        st.write(f" {entry['answer']}")
        st.write("---")

# Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    clear_chroma_db()  # Also clear the Chroma DB when the button is clicked






# '''code4.1'''
# import streamlit as st
# import sys
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# # from langchain.vectorstores import DocArrayInMemorySearch

# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain.prompts.chat import MessagesPlaceholder

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter
# from langchain_community.vectorstores import Chroma  # Only import once

# # Saving conversation history with this code
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory

# # Download NLTK punkt if needed
# import nltk
# nltk.download('punkt')  # Correct the download name



# import os
# # Load Google API Key from environment variable for security
# google_api_key = os.getenv("GOOGLE_API_KEY")

# # Initialize the model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=google_api_key)


# # Set up memory and conversation chain
# memory = ConversationBufferMemory()
# conversation_chain = ConversationChain(llm=model, memory=memory)


# # PDF loading and splitting
# pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # Embed the chunks and store them in a vector store
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# # Create Chroma vectorstore and persist it
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
# db.persist()

# # Connect to the Chroma vectorstore
# db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# # Create the retriever
# retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# # Create chat template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""),
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information. You are a personal assistant for the Remark Job And Recruiter Portal. 
#     Context: {context}

#     Question: {question}
#     Answer: """)
# ])

# # Create document chain and output parser
# document_chain = create_stuff_documents_chain(model, chat_template)
# output_parser = StrOutputParser()

# # Chain for retrieval and answering
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# # Streamlit app setup
# st.title("ChatBot")

# # Initialize session state for storing chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Function to delete the entire Chroma collection
# def clear_chroma_db():
#     # Load the vectorstore
#     db = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
#     # Delete the entire collection
#     db.delete_collection()
#     st.success("Deleted ")

# # Function to handle chat and store response
# def chat_with_remark(user_input):
#     # Check if user input is 'clear' and clear the Chroma database
#     if user_input.lower() == "clear":
#         clear_chroma_db()  # Clear Chroma DB
#         return "Chroma DB and chat history have been cleared."

#     all_chunks = []
#     for chunk in rag_chain.stream(user_input):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

#     return ''.join(all_chunks)

# # Input for user question
# user_input = st.text_input("Ask a Question:", key="user_input")

# # If there's user input, generate a response
# if user_input:
#     answer = chat_with_remark(user_input)
#     st.session_state.chat_history.append({"question": user_input, "answer": answer})

# # Display chat history
# if st.session_state.chat_history:
#     st.write("### Chat")
#     for entry in st.session_state.chat_history:
#         st.write(f" {entry['question']}")
#         st.write(f" {entry['answer']}")
#         st.write("---")

# # Button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     clear_chroma_db()  # Also clear the Chroma DB when the button is clicked





from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from some_module import rag_chain  # Replace with the actual import for your chain
import sys

# Initialize memory with a window size of 3
memory = ConversationBufferWindowMemory(k=3, return_messages=True)



# '''code 6'''

# import streamlit as st
# import sys
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# # from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain.prompts.chat import MessagesPlaceholder

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter
# from langchain_community.vectorstores import Chroma  # Only import once

# # Saving conversation history with this code
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory

# # Download NLTK punkt if needed
# import nltk
# nltk.download('punkt')  # Correct the download name



# import os
# # Load Google API Key from environment variable for security
# google_api_key = os.getenv("GOOGLE_API_KEY")

# # Initialize the model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=google_api_key)


# # Set up memory and conversation chain
# memory = ConversationBufferWindowMemory(k=3, return_messages=True)
# conversation_chain = ConversationChain(llm=model, memory=memory)


# # PDF loading and splitting
# pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # Embed the chunks and store them in a vector store
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# # Create Chroma vectorstore and persist it
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
# db.persist()

# # Connect to the Chroma vectorstore
# db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# # Create the retriever
# retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# # Create chat template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""),
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information. You are a personal assistant for the Remark Job And Recruiter Portal. 
#     Context: {context}

#     Question: {question}
#     Answer: """)
# ])

# # Create document chain and output parser
# document_chain = create_stuff_documents_chain(model, chat_template)
# output_parser = StrOutputParser()

# # Chain for retrieval and answering
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# # Streamlit app setup
# st.title("ChatBot")

# # Initialize session state for storing chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []

# # Function to delete the entire Chroma collection
# def clear_chroma_db():
#     # Load the vectorstore
#     db = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
#     # Delete the entire collection
#     db.delete_collection()
#     st.success("Deleted ")

# # Function to handle chat and store response
# def chat_with_remark(user_input):
#     # Check if user input is 'clear' and clear the Chroma database
#     if user_input.lower() == "clear":
#         clear_chroma_db()  # Clear Chroma DB
#         return "Chroma DB and chat history have been cleared."

#     all_chunks = []
#     for chunk in rag_chain.stream(user_input):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

#     return ''.join(all_chunks)

# # Input for user question
# user_input = st.text_input("Ask a Question:", key="user_input")

# # If there's user input, generate a response
# if user_input:
#     answer = chat_with_remark(user_input)
#     st.session_state.chat_history.append({"question": user_input, "answer": answer})

# # Display chat history
# if st.session_state.chat_history:
#     st.write("### Chat")
#     for entry in st.session_state.chat_history:
#         st.write(f" {entry['question']}")
#         st.write(f" {entry['answer']}")
#         st.write("---")

# # Button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     clear_chroma_db()  # Also clear the Chroma DB when the button is clicked








# '''code 7'''


# from collections import deque
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain_core.messages import BaseMessage

# class FixedSizeConversationBufferMemory(ConversationBufferMemory):
#     def __init__(self, k=5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.window_size = k
#         self.messages = deque(maxlen=k)

#     def append_message(self, message: BaseMessage):
#         self.messages.append(message)
#         super().append_message(message)

#     def get_memory(self):
#         return list(self.messages)



# import streamlit as st
# import sys
# import json
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain_core.messages import SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import NLTKTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.chains import ConversationChain
# from langchain_core.runnables import (
#     RunnableLambda,
#     RunnableParallel,
#     RunnablePassthrough,
# )
# import nltk
# nltk.download('punkt')

# # Custom FixedSizeConversationBufferMemory class
# from collections import deque
# from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain_core.messages import BaseMessage

# class FixedSizeConversationBufferMemory(ConversationBufferMemory):
#     def __init__(self, k=5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.window_size = k
#         self.messages = deque(maxlen=k)

#     def append_message(self, message: BaseMessage):
#         self.messages.append(message)
#         super().append_message(message)

#     def get_memory(self):
#         return list(self.messages)

# # Load Google API Key from environment variable for security
# google_api_key = os.getenv("GOOGLE_API_KEY")

# # Initialize the model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest', temperature=0.7, google_api_key=google_api_key)

# # Set up memory with a fixed size of 5 messages
# memory = FixedSizeConversationBufferMemory(window_size=5)
# conversation_chain = ConversationChain(llm=model, memory=memory)

# # PDF loading and splitting
# pdf_path = r"C:\python\Updated Remark App Description.pdf"  # Modify this to your path
# loader = PyPDFLoader(pdf_path)
# documents = loader.load()

# # Split the content using NLTKTextSplitter
# text_splitter = NLTKTextSplitter(chunk_size=2000, chunk_overlap=100)
# chunks = text_splitter.split_documents(documents)

# # Embed the chunks and store them in a vector store
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# # Create Chroma vectorstore and persist it
# db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma")
# db.persist()

# # Connect to the Chroma vectorstore
# db_connection = Chroma(persist_directory="./chroma", embedding_function=embedding_model)

# # Create the retriever
# retriever = db_connection.as_retriever(search_kwargs={"k": 9})

# # Create chat template
# chat_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content="""You are a personal assistant for user and your name is Remark. Given a context and question from the user, you should answer based on the given context and ask for more information if needed."""), 
#     HumanMessagePromptTemplate.from_template("""Answer the question based on the given context and ask if needed the more information. You are a personal assistant for the Remark Job And Recruiter Portal. 
#     Context: {context}

#     Question: {question}
#     Answer: """)
# ])

# # Create document chain and output parser
# document_chain = create_stuff_documents_chain(model, chat_template)
# output_parser = StrOutputParser()

# # Chain for retrieval and answering
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | chat_template
#     | model
#     | output_parser
# )

# # File path for chat history
# CHAT_HISTORY_FILE = "chat_history.json"

# # Function to load chat history from a JSON file
# def load_chat_history():
#     if os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, "r") as file:
#             return json.load(file)
#     return []

# # Function to save chat history to a JSON file
# def save_chat_history(chat_history):
#     with open(CHAT_HISTORY_FILE, "w") as file:
#         json.dump(chat_history, file)

# # Streamlit app setup
# st.title("ChatBot")

# # Initialize session state for storing chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = load_chat_history()

# # Function to delete the entire Chroma collection
# def clear_chroma_db():
#     # Load the vectorstore
#     db = Chroma(persist_directory="./chroma", embedding_function=embedding_model)
#     # Delete the entire collection
#     db.delete_collection()
#     st.success("Deleted ")

# # Function to handle chat and store response
# def chat_with_remark(user_input):
#     # Check if user input is 'clear' and clear the Chroma database
#     if user_input.lower() == "clear":
#         clear_chroma_db()  # Clear Chroma DB
#         return "Chroma DB and chat history have been cleared."

#     all_chunks = []
#     for chunk in rag_chain.stream(user_input):
#         all_chunks.append(chunk)
#         sys.stdout.write(chunk)
#         sys.stdout.flush()

#     response = ''.join(all_chunks)
#     return response

# # Input for user question
# user_input = st.text_input("Ask a Question:", key="user_input")

# # If there's user input, generate a response
# if user_input:
#     answer = chat_with_remark(user_input)
#     st.session_state.chat_history.append({"question": user_input, "answer": answer})
#     save_chat_history(st.session_state.chat_history)  # Save chat history to file

# # Display chat history
# if st.session_state.chat_history:
#     st.write("### Chat")
#     for entry in st.session_state.chat_history:
#         st.write(f"**Question:** {entry['question']}")
#         st.write(f"**Answer:** {entry['answer']}")
#         st.write("---")

# # Button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     save_chat_history(st.session_state.chat_history)  # Save cleared history
#     clear_chroma_db()  # Also clear the Chroma DB when the button is clicked















