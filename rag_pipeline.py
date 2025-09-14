#Aqui ficará toda a lógica de carregar os currículos (documentos), 
# dividi-los em pedaços (chunks) e criar o sistema de busca (retriever) 
# que o agente usará para encontrar informações.

import re
import pathlib
from typing import List, Dict
from pathlib import Path

# Importações da biblioteca LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Importa a chave de API do nosso arquivo de configuração
from config import GOOGLE_API_KEY

# --- 1. Carregamento e Processamento dos Documentos ---

def carregar_documentos(caminho_pasta: str) -> List:
    """Carrega todos os arquivos PDF de uma pasta."""
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
    """Divide os documentos em chunks menores."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(documentos)

# --- 2. Criação do Retriever (Base de Conhecimento Vetorial) ---

def criar_retriever(chunks: List, api_key: str):
    """Cria um retriever a partir dos chunks de texto."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )
    
# --- 3. Criação da Chain de Pergunta e Resposta com RAG ---

def criar_rag_chain(api_key: str):
    """Cria a chain que combina o LLM com o prompt RAG."""
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=api_key
    )
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
         "Você é um Assistente para usuarios que entrem no portifolio da programadora Lizandra Santos.\n"
         "Responda SOMENTE com base no contexto fornecido.\n"
         "Se não houver contexto suficiente para responder a pergunta, responda 'Não sei'.\n"),
        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
    ])
    return create_stuff_documents_chain(llm, prompt_rag)

# --- 4. Funções de Formatação de Citações ---

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
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

# --- 5. Função Principal para Perguntas RAG (Junta tudo) ---

# Carregamos e processamos os documentos uma única vez quando o módulo é importado
documentos = carregar_documentos("./curriculos")
chunks = dividir_documentos(documentos)
retriever = criar_retriever(chunks, GOOGLE_API_KEY)
document_chain = criar_rag_chain(GOOGLE_API_KEY)


def perguntar_informacoes_RAG(pergunta: str) -> Dict:
    """Função que executa a busca e geração de resposta."""
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
