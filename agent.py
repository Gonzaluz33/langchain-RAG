import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from database import search_db_by_keywords
from vectorstore import get_vectorstore
from langchain import SQLDatabase

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("La variable de entorno 'ANTHROPIC_API_KEY' no está configurada.")

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    max_tokens=1024,
    timeout=None,
    max_retries=2,
    anthropic_api_key=anthropic_api_key
)

keyword_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    Extrae exactamente 3 palabras clave más importantes de la siguiente consulta y sepáralas por comas sin espacios adicionales (centrate en las tecnologías):
    Consulta: {query}

    Ejemplo1:
    Consulta: "Necesito un asistente para pruebas automatizadas en JIRA"
    Respuesta: "JIRA,pruebas,automatizadas"
    
    Ejemplo2:
    Consulta: "Necesito un asistente para Angular"
    Respuesta: "angular,frontend,codigo"
    
    Ejemplo3: 
    consulta: "Necesito un asistente para documentar mi codigo java"
    Respuesta: "java,documentacion,codigo"
    """.strip()
)

keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt)

def search_vectorstore(query: str):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=5)
    return [d.page_content for d in docs]

search_db_tool = Tool(
    name="search_db",
    func=search_db_by_keywords,
    description="Usar esta herramienta para buscar asistentes en la base de datos relacional usando keywords extraidas."
)

vectorstore_tool = Tool(
    name="search_vectorstore",
    func=search_vectorstore,
    description="Usar esta herramienta para buscar asistentes en la base vectorial dado un query."
)

agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=
        """
        Eres un agente que ayuda a seleccionar el mejor asistente de IA. La secuencia recomendada:

        1. Extrae las palabras clave de la consulta del usuario (ya se hizo y se pasará en el input).
        2. Usa 'search_db' con esas palabras clave para ver si hay asistentes relevantes en la base relacional.
        3. Usa 'search_vectorstore' con la consulta original para obtener asistentes vectoriales relevantes.
        4. Combina los resultados y da la mejor recomendación.

        Consulta del usuario: {input}

        {agent_scratchpad}
        """
)

tools = [search_db_tool, vectorstore_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    prompt=agent_prompt
)
