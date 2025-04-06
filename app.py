import streamlit as st
import fitz  # PyMuPDF para extração de texto
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os

# Configuração da API da OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Modelo de embeddings
modelo = SentenceTransformer("all-MiniLM-L6-v2")

# Função para extrair texto de um PDF
def extrair_texto_pdf(caminho_pdf):
    doc = fitz.open(caminho_pdf)
    texto = "\n".join([page.get_text() for page in doc])
    return texto

# Função para criar embeddings e indexar os textos
def criar_indice(textos):
    embeddings = modelo.encode(textos, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Função para buscar a resposta no índice
def buscar_resposta(pergunta, textos, index, embeddings):
    emb_pergunta = modelo.encode([pergunta], convert_to_numpy=True)
    _, indices = index.search(emb_pergunta, 1)
    texto_relevante = textos[indices[0][0]]

    resposta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Responda com base no seguinte texto: " + texto_relevante},
            {"role": "user", "content": pergunta}
        ]
    )
    
    return resposta["choices"][0]["message"]["content"]

# Interface com Streamlit
st.title("Chatbot baseado em PDFs 📄🤖")

uploaded_file = st.file_uploader("Faça upload do seu PDF", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    texto = extrair_texto_pdf("temp.pdf")
    textos = texto.split("\n")
    index, embeddings = criar_indice(textos)

    pergunta = st.text_input("Digite sua pergunta:")
    if st.button("Buscar Resposta") and pergunta:
        resposta = buscar_resposta(pergunta, textos, index, embeddings)
        st.write("Resposta:", resposta)