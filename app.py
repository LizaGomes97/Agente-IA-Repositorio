from fastapi import FastAPI
from pydantic import BaseModel
from agent_workflow import grafo  # Importa o grafo compilado
from fastapi.middleware.cors import CORSMiddleware

# Cria a aplicação FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite todas as origens (para desenvolvimento)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------

# Define o formato do corpo da requisição (o que o front-end vai enviar)
class ChatRequest(BaseModel):
    pergunta: str
    
# Define a rota principal da nossa API
@app.post("/chat")
def conversar_com_agente(request: ChatRequest):
    """
    Recebe uma pergunta, invoca o agente e retorna a resposta.
    """
    # A pergunta vem do corpo da requisição
    pergunta_usuario = request.pergunta

    # Invoca o grafo do langgraph com o estado inicial
    resposta_final = grafo.invoke({"pergunta": pergunta_usuario})

    # Retorna a resposta final e as citações para o front-end
    return {
        "resposta": resposta_final.get("resposta"),
        "citacoes": resposta_final.get("citacoes", [])
    }

# Para rodar o servidor localmente, use o comando no terminal:
# uvicorn app:app --reload