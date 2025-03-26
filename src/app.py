import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb.config import Settings
import os

load_dotenv()

# Website Scraping
def get_vectorstore_from_url(url):
    # gets the text from the website in the document form
    loader = WebBaseLoader(url)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    # create a vector store with Chroma in memory to avoid tenant errors
    vector_store = Chroma.from_documents(
        document_chunks, 
        OpenAIEmbeddings()
    )
    return vector_store

# It is used to get the context from vector store
def get_context_retrieval_chain(vector_store):
    # Here we will be embedding all the query (entire convo history) from the Human and retrieve it from vector database
    # And we will use it to get the chunks of text relevant to the conversation
    
    # it will be organized in a pipeline way
    llm = ChatOpenAI()
    
    # in langchain, as_retriever is used to retrieve information regarding context
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        # message placeholder loads the entire chat history
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

# it uses the documents related to the chat to generate answer
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    # this combines the context and the user input (combined document chain)
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    #create retrieval chain combines the document context and user query and provide us answer 
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)  

def get_response(user_input):
    retriever_chain = get_context_retrieval_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# App config
st.set_page_config(page_title= 'Chat with Website', page_icon= '🛸')

st.title('Chat with Website')


# Sidebar for the website URL
with st.sidebar:
    st.header("Website")
    website_url = st.text_input('Enter website URL:')


if website_url is None or website_url == '':
    st.info("Please enter a website URL")
else:
    
    # session_state helps to retain this memory whenever streamlit application has some change in it
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello!! I am a bot, How can I help you?')
            ]
    
    # create conversation chain
    # the vector store should be consistent, so new embeddings are not made whenever we make a query
    if 'vector_store' not in st.session_state:
        
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    
    

    # Chatbot user-input
    user_query = st.chat_input("Type your message here")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Conversation
    for message in st.session_state.chat_history:
        
        if isinstance(message, AIMessage):
            
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            
            with st.chat_message('Human'):
                st.write(message.content)
                
