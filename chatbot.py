import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from ingestData import initialize_vector_store

def create_rag_system(vector_store, gemini_api_key):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retriever_tool = create_retriever_tool(
        retriever,
        "data_search",
        "Search for information in the vector database and return the most relevant information in as much detail as possible."
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    llm_with_tools = llm.bind_tools([retriever_tool])
    return llm_with_tools

def main():
    st.title("Query with Gemini LLM")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    db_token = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    db_id = st.secrets["ASTRA_DB_ID"]
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    
    vector_store = initialize_vector_store(db_token, db_id)
    llm_with_tools = create_rag_system(vector_store, gemini_api_key)

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    query = st.chat_input("Ask me anything:", key="user_input")
    if query:
        st.session_state.chat_history.append(("user", query))
        st.chat_message("user").markdown(query)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Ensure the context is provided
        full_prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer based only on the above context."
        response = llm_with_tools.invoke(full_prompt)
        response_content = response.content.replace('\n', '<br>')
        st.chat_message("assistant").markdown(AIMessage(response_content), unsafe_allow_html=True)
        st.session_state.chat_history.append(("assistant", response))

        if len(st.session_state.chat_history) > 25:
            st.session_state.chat_history = st.session_state.chat_history[-25:]
        
        # Display tool calls if any
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        if tool_calls:
            for tool_call in tool_calls:
                st.write(f"Tool call: {tool_call}")

    # Scroll to the bottom to show the latest message
    st.write("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
