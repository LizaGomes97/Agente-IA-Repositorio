#Aqui ficará toda a lógica de carregar os currículos (documentos), 
# dividi-los em pedaços (chunks) e criar o sistema de busca (retriever) 
# que o agente usará para encontrar informações.

import re
import pathlib
import os
from typing import List, Dict
from pathlib import Path
import time

# Importações da biblioteca LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint 
from config import GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN

# Importa a chave de API do nosso arquivo de configuração
from config import GOOGLE_API_KEY

# --- NOME DO ARQUIVO PARA SALVAR O ÍNDICE FAISS ---
FAISS_INDEX_PATH = "faiss_index"

# --- 1. Funções de Carregamento e Processamento (sem alterações) ---

def carregar_documentos(caminho_pasta: str) -> List:
    # ... (esta função continua exatamente a mesma)
    pasta = Path(caminho_pasta)
    curriculos = []
    for arquivo_pdf in pasta.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(arquivo_pdf))
            curriculos.extend(loader.load())
            print(f"Arquivo '{arquivo_pdf.name}' carregado com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar o arquivo {arquivo_pdf.name}: {e}")
    return curriculos

def dividir_documentos(documentos: List) -> List:
    # ... (esta função continua exatamente a mesma)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(documentos)

# --- 2. Criação OU Carregamento do Retriever (LÓGICA MODIFICADA) ---

def obter_ou_criar_retriever(chunks: List, api_key: str):
    """
    Carrega o retriever do disco se existir, senão, cria um novo em lotes para respeitar os limites da API.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print("Carregando base de conhecimento local (FAISS index)...")
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Criando uma nova base de conhecimento (FAISS index) em lotes...")
        # Lógica para criar o índice em lotes para evitar erros de quota
        vectorstore = None
        # Processa 100 chunks de cada vez
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            if vectorstore is None:
                # Cria o vectorstore com o primeiro lote
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                # Adiciona os lotes seguintes ao vectorstore existente
                vectorstore.add_documents(batch)
            
            print(f"Lote {i//100 + 1} processado. Aguardando 1 segundo...")
            time.sleep(1) # <--- A MÁGICA ACONTECE AQUI!

        if vectorstore:
            vectorstore.save_local(FAISS_INDEX_PATH)
            print(f"Base de conhecimento salva em '{FAISS_INDEX_PATH}'")

    if not vectorstore:
        raise Exception("Falha ao criar ou carregar a base de conhecimento.")

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )

# --- 3 & 4. Chain RAG e Funções de Formatação (sem alterações) ---

def criar_rag_chain(): 
    """Cria a chain que combina o LLM com o prompt RAG."""
    # 3. Configuramos o endpoint do Gemma
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-9b-it", # Usando o modelo Gemma
        max_length=1024,
        temperature=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um Assistente para usuarios que entrem no portifolio da programadora Lizandra Santos.\n"
         "Responda SOMENTE com base no contexto fornecido.\n"
         "Se não houver contexto suficiente para responder a pergunta, responda 'Não sei'.\n"),
        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])
    return create_stuff_documents_chain(llm, prompt_rag)

def _clean_text(s: str) -> str:
    # ... (esta função continua exatamente a mesma)
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    # ... (esta função continua exatamente a mesma)
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    # ... (esta função continua exatamente a mesma)
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source", "")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

# --- 5. Função Principal e Inicialização (LÓGICA MODIFICADA) ---

# Carregamos e processamos os documentos
documentos = carregar_documentos("./curriculos")
chunks = dividir_documentos(documentos)

# OBTEMOS o retriever (carregando ou criando)
retriever = obter_ou_criar_retriever(chunks, GOOGLE_API_KEY)
# Cria a chain RAG com o novo LLM (Gemma)
document_chain = criar_rag_chain() 

def perguntar_informacoes_RAG(pergunta: str) -> Dict:
    # ... (esta função continua exatamente a mesma)
    docs = retriever.get_relevant_documents(pergunta)

    if not docs:
        return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta, "context": docs})
    txt = (answer or '').strip()

    if txt.rstrip('.!?') == "Não sei":
        return {"answer": "Não sei", "citacoes": [], "contexto_encontrado": False}

    return {
        "answer": txt,
        "citacoes": formatar_citacoes(docs, pergunta),
        "contexto_encontrado": True
    }