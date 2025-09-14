#Estrutura Inicial

/agente-ia-portfolio/
|
├── curriculos/
|   └── seu_curriculo.pdf
|
├── config.py             # Chaves de API e configurações
├── rag_pipeline.py       # Toda a lógica de RAG (carregar PDF, criar chunks, retriever)
├── agent_workflow.py     # Definição dos nós e do grafo com langgraph
└── app.py                # O servidor web (API) que vai expor o agente


config.py -> Guarda a GOOGLE_API_KEY
//chave da API

rag_pipeline.py -> Carrega curriculos

agent_workflow.py -> Definições do AgentState

app.py -> Ponto de entrada. Importa o grafo do agent_workflow.py.

[![Notebook do codigo base no Google Colab](https://colab.research.google.com/img/colab_favicon_256px.png)](https://github.com/LizaGomes97/Agente-IA-Portifolio.git) 

