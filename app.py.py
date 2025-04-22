
import streamlit as st
import json
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Membaca API key dari Streamlit Secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load data FAQ dari JSON
with open("ConsultaxAI_FAQ_PPh_Orang_Pribadi_130.json", encoding="utf-8") as f:
    faq_data = json.load(f)

docs = [
    Document(
        page_content=f"Q: {item['question']}\nA: {item['answer']}",
        metadata={
            "category": item.get("category", ""),
            "source": item.get("source", "")
        }
    )
    for item in faq_data
]

# Buat vectorstore dan retriever
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# Buat QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

# Streamlit UI
st.set_page_config(page_title="ConsultaxAI", page_icon="💬")
st.title("🤖 ConsultaxAI – Konsultan Pajak AI")
st.markdown("Tanyakan apa pun seputar **PPh**")

query = st.text_input("Pertanyaan Anda")

if query:
    with st.spinner("Sedang mencari jawaban..."):
        response = qa.run(query)
        st.markdown("### 💡 Jawaban:")
        st.write(response)
