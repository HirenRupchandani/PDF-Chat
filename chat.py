import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from streamlit_chat import message

load_dotenv()
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def ingest_docs(pdf_path: str):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_pdf_index_react")
    print("****Loading to vectorstore done ***")


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    new_vectorstore = FAISS.load_local("faiss_pdf_index_react", embeddings, allow_dangerous_deserialization=True)
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=new_vectorstore.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


def ingest_pdf(file):
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        ingest_docs(tmp_file_path)
        st.success("PDF successfully ingested!")


def chat_with_pdf():
    st.title("PDF Question-Answering System")
    chat_placeholder = st.empty()
    form_container = st.container()

    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0

    uploaded_file = form_container.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        ingest_pdf(uploaded_file)

    if (
            "chat_answers_history" not in st.session_state
            and "user_prompt_history" not in st.session_state
            and "chat_history" not in st.session_state
    ):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

    # Display form only if the query count is less than 10
    if st.session_state.query_count < 10:
        with form_container.form(key='input_form', clear_on_submit=True):
            prompt = st.text_input("Prompt", placeholder="e.g., Tell me something about this document")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and prompt:
            st.session_state.query_count += 1
            with st.spinner("Generating response..."):
                generated_response = run_llm(
                    query=prompt, chat_history=st.session_state["chat_history"]
                )

                response_text = generated_response["answer"]
                st.session_state.chat_history.append((prompt, response_text))
                st.session_state.user_prompt_history.append(prompt)
                st.session_state.chat_answers_history.append(response_text)
    else:
        form_container.write("Input is gone. You are out of allowed query limits for the session")

    with chat_placeholder.container():
        if st.session_state["chat_answers_history"]:
            for generated_response, user_query in zip(
                    st.session_state["chat_answers_history"],
                    st.session_state["user_prompt_history"],
            ):
                message(user_query, is_user=True)
                message(generated_response)


if __name__ == "__main__":
    chat_with_pdf()
