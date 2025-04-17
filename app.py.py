
import streamlit as st
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Set API Key OpenAI Anda
os.environ["OPENAI_API_KEY"] = "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"

# Load data FAQ
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

# Buat FAISS vector store
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=retriever
)

# Streamlit UI
st.set_page_config(page_title="ConsultaxAI", page_icon="💬")
st.title("🤖 ConsultaxAI – Konsultan Pajak AI")
st.markdown("Tanyakan apa pun seputar **PPh Orang Pribadi**, berdasarkan FAQ dan peraturan yang tersedia.")

query = st.text_input("Pertanyaan Anda")

if query:
    with st.spinner("Sedang memproses jawaban..."):
        answer = qa.run(query)
        st.markdown("### 💡 Jawaban:")
        st.write(answer)
Delete app_consultaxai.py (replaced by app.py)
