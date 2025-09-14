from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Importações da biblioteca LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Importa a função RAG e a chave de API
from rag_pipeline import perguntar_informacoes_RAG
from langchain_huggingface import HuggingFaceEndpoint
from config import GOOGLE_API_KEY, HUGGINGFACEHUB_API_TOKEN

# --- 1. Definição do Estado do Agente e Modelos de Saída ---

class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ENVIAR_EMAIL"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    detalhes: List[str] = Field(default_factory=list)

# --- 2. Configuração do LLM para Triagem ---

TRIAGEM_PROMPT = """
    Voce é um assistente do portifolio de Lizandra Ribeiro Gomes Placido dos Santos e seu objetivo é ajudar os usuarios que entram no portifolio a encontrar o que buscam e tirar suas duvidas, no fim do atendimento voce sempre vai informar a Lizandra tudo o que aconteceu em sua conversa.
    Lizandra tem sua propria empresa a CodeStorm na qual ela realiza trabalhos autonomos e ela tambem tem outras experiencias (veja no curriculo)
    Dada a mensagem do usuario, retorne SOMENTE um JSON com:
    {
      "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ENVIAR_EMAIL",
      "urgencia": "BAIXA" | "MEDIA" | "ALTA",
      "detalhes": ["..."]
    }
    Regras:
    - **AUTO_RESOLVER**: Perguntas claras sobre curriculo, habilidades, certificados, projetos ou forma de entrar em contato descritas nos curriculos. (Ex: "Voce possui habilidade com Python?", "Qual seu nivel de habilidade com Front-End?").
    - **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou o contexto (Ex: "Tenho uma duvida", "Quero informações")
    - **ENVIAR_EMAIL**: Após encerrar o atendimento do usuario ou quando o usuario pede para entrar em contato com Lizandra (Ex:"Gostaria de conversar com você", "Tenho uma proposta para voce")
"""

llm_triagem = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    max_length=256, # Menor para a tarefa de triagem
    temperature=0,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

# --- 3. Definição dos Nós do Grafo ---

def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó: triagem...")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó: auto_resolver...")
    resposta_rag = perguntar_informacoes_RAG(state["pergunta"])
    update = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó: pedir_info...")
    detalhe = "tema e contexto específico"
    return {
        "resposta": f"Para avançar, preciso de mais informações sobre {detalhe}.",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_enviar_email(state: AgentState) -> AgentState:
    print("Executando nó: enviar_email...")
    return {
        "resposta": f"Entendido! Enviarei um email para a Lizandra com sua mensagem para que ela possa entrar em contato. Descrição: '{state['pergunta'][:140]}'",
        "citacoes": [],
        "acao_final": "ENVIAR_EMAIL"
    }

# --- 4. Definição das Arestas Condicionais (Lógica de Fluxo) ---

KEYWORDS_ABRIR_TICKED = ["contate-me", "conversar", "comigo", "proposta", "conectar"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]
    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    return "email"

def decidir_pos_autoresolver(state: AgentState) -> str:
    print("Decidindo após auto_resolver...")
    if state.get("rag_sucesso"):
        return "ok"
    if any(k in state["pergunta"].lower() for k in KEYWORDS_ABRIR_TICKED):
        return "email"
    return "info"

# --- 5. Construção e Compilação do Grafo ---

workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("enviar_email", node_enviar_email)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "email": "enviar_email"
})
workflow.add_conditional_edges("auto_resolver", decidir_pos_autoresolver, {
    "info": "pedir_info",
    "email": "enviar_email",
    "ok": END
})
workflow.add_edge("pedir_info", END)
workflow.add_edge("enviar_email", END)

grafo = workflow.compile()
