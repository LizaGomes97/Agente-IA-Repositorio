import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente de um arquivo .env
# (Isso é opcional, mas altamente recomendado para segurança)
load_dotenv()

# Pega a chave da API do ambiente.
# Certifique-se de ter um arquivo .env ou de configurar a variável de ambiente no seu sistema.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Se a chave não for encontrada, você pode descomentar a linha abaixo
# e colocar sua chave diretamente, mas evite fazer isso em repositórios públicos.
# GOOGLE_API_KEY = "SUA_CHAVE_API_AQUI"
