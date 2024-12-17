import os
import pandas as pd
from typing import List
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Float, ForeignKey, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_anthropic import ChatAnthropic

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain import SQLDatabase
from sqlalchemy import text

# Watsonx imports
from ibm_watsonx_ai import Credentials, APIClient
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

load_dotenv()

# --------------------------
# Configuración de la BD Postgres (FUENTE DE VERDAD)
# --------------------------
POSTGRES_USER = os.getenv("DB_USER")
POSTGRES_PASSWORD = os.getenv("DB_PASSWORD")
POSTGRES_HOST = os.getenv("DB_HOST", "localhost")
POSTGRES_PORT = os.getenv("DB_PORT", "5432")
POSTGRES_DB = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

Base = declarative_base()

class Asistente(Base):
    __tablename__ = 'asistentes'
    id = Column(Integer, primary_key=True)
    nombre = Column(String(255), nullable=False)
    descripcion = Column(Text)
    input_recomendado = Column(Text)
    output_esperado = Column(Text)
    tecnologias = Column(ARRAY(String))
    categoria = Column(String(100))
    keywords = Column(ARRAY(String))
    url_repo_pruebas = Column(String(255))
    estado_pruebas = Column(String(50))
    comentarios = Column(Text)
    pruebas = relationship("Prueba", backref="asistente")

class Prueba(Base):
    __tablename__ = 'pruebas'
    id = Column(Integer, primary_key=True)
    asistente_id = Column(Integer, ForeignKey('asistentes.id'), nullable=False)
    caso_uso = Column(Text)
    descripcion = Column(Text)
    input_dado = Column(Text)
    output_generado = Column(Text)
    correcciones_necesarias = Column(Text)
    metrica_eficiencia = Column(Float) # Tiempo ahorado en porcentaje (0-100)
    metrica_precision = Column(Float) # Que tan acertada fue la respuesta sin hacer correciónes (0-100)
    resumen = Column(Text)
    conclusion = Column(Text)
    fecha_ejecucion = Column(Date)

class Categoria(Base):
    __tablename__ = 'categorias'
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)

engine = create_engine(DATABASE_URL, echo=False)
# Después de crear engine, puedes crear un SQLDatabase LangChain:
db = SQLDatabase.from_uri(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

# --------------------------
# Embeddings con Ollama
# --------------------------
class OllamaEmbeddings:
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for idx, t in enumerate(texts):
            print(f"Generando embedding para asistente {idx + 1}/{len(texts)}...")
            try:
                response = ollama.embeddings(model=self.model_name, prompt=t)
                embedding = response.get("embedding")
                if embedding and len(embedding) > 0:
                    embeddings.append(embedding)
                else:
                    print(f"Embedding vacío o inválido para el asistente {idx + 1}")
                    embeddings.append([])
            except Exception as e:
                print(f"Error al generar embedding para el asistente {idx + 1}: {e}")
                embeddings.append([])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response.get("embedding")
            if embedding and len(embedding) > 0:
                return embedding
            else:
                return []
        except Exception as e:
            print(f"Error al generar embedding para la consulta: {e}")
            return []

class OllamaLangchainEmbeddings(Embeddings):
    def __init__(self, ollama_embeddings: OllamaEmbeddings):
        self.ollama_embeddings = ollama_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ollama_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.ollama_embeddings.embed_query(text)

ollama_model = OllamaEmbeddings("mxbai-embed-large")
lc_embeddings = OllamaLangchainEmbeddings(ollama_model)

# --------------------------
# Helper: Crear texto desde un asistente para Chroma
# --------------------------
def asistente_to_text(asistente: Asistente) -> str:
    tech_list = asistente.tecnologia if asistente.tecnologia else []
    return (
        f"Nombre: {asistente.nombre}\n"
        f"Description: {asistente.descripcion or ''}\n"
        f"Inputs: {asistente.input_recomendado or ''}\n"
        f"Outputs: {asistente.output_esperado or ''}\n"
        f"Technology: {', '.join(tech_list)}\n"
        f"Category: {asistente.categoria or ''}\n"
    )

# --------------------------
# Función para obtener vectorstore
# Se usa la misma colección y directorio persistente
# --------------------------
def get_vectorstore() -> Chroma:
    vectorstore = Chroma(
        collection_name="asistentes_collection",
        persist_directory="chroma_db",
        embedding_function=lc_embeddings
    )
    return vectorstore

# --------------------------
# Cargar desde Excel a Postgres y Chroma
# --------------------------
def load_assistants_from_excel(excel_path: str):
    df = pd.read_excel(excel_path)

    assistants_data = []
    nuevos_asistentes = []
    for idx, row in df.iterrows():

        # Parsear las tecnologías
        technology_str = str(row.get("Technology", ""))
        if technology_str.strip():
            tecnologia_list = [t.strip() for t in technology_str.split("/") if t.strip()]
        else:
            tecnologia_list = []

        nombre = str(row.get("GenAI_Asset", ""))
        description = str(row.get("Description", ""))
        url_repo_pruebas = str(row.get("Test_Cases", ""))
        input_data = str(row.get("Input", ""))
        expected_output = str(row.get("Expected_Output", ""))
        category = str(row.get("Category", ""))

        # Parsear las keywords
        keywords_str = str(row.get("Keywords", ""))
        if keywords_str.strip():
            keywords_list = [k.strip() for k in keywords_str.split(",") if k.strip()]
        else:
            keywords_list = []

        existing_asistente = session.query(Asistente).filter_by(nombre=nombre).first()
        if existing_asistente:
            print(f"Ya existe un asistente con el nombre '{nombre}' en Postgres. No se agregará de nuevo.")
            continue
      
        nuevo_asistente = Asistente(
            nombre=nombre,
            descripcion=description,
            input_recomendado=input_data,
            output_esperado=expected_output,
            tecnologias=tecnologia_list,
            categoria=category,
            keywords=keywords_list,
            url_repo_pruebas=url_repo_pruebas,
            estado_pruebas=None,
            comentarios=None
        )
        session.add(nuevo_asistente)
        nuevos_asistentes.append(nuevo_asistente)

        texto_final = (
            f"Nombre: {nombre}\n"
            f"Asset_ID: {nuevo_asistente.id}\n"
            f"Technology: {technology_str}\n"
            f"Description: {description}\n"
            f"Test Cases: {url_repo_pruebas}\n"
            f"Input: {input_data}\n"
            f"Expected Output: {expected_output}\n"
            f"Keywords: {keywords_str}\n"
            f"Category: {category}\n"
        )
        assistants_data.append(texto_final)

    session.commit()
    print("Asistentes cargados en Postgres desde Excel.")

    # Ahora actualizar el vectorstore
    vectorstore = get_vectorstore()
    for a, t in zip(nuevos_asistentes, assistants_data):
        vectorstore.add_texts(
            texts=[t],
            ids=[str(a.id)],
            metadatas=[{"id": a.id}]
        )
    # No es necesario llamar a persist()
    print("Asistentes agregados a Chroma.")

# --------------------------
# CRUD: Agregar asistente manualmente
# --------------------------
def add_asistente_manualmente(
    nombre: str,
    descripcion: str,
    input_recomendado: str,
    output_esperado: str,
    tecnologias: List[str],
    categoria: str,
    url_repo_pruebas: str = None,
    fecha_prueba: date = None,
    estado_prueba: str = None,
    comentarios: str = None
):
    existing_asistente = session.query(Asistente).filter_by(nombre=nombre).first()
    if existing_asistente:
        print(f"Ya existe un asistente con el nombre '{nombre}' en Postgres. No se agregará de nuevo.")
        return

    nuevo_asistente = Asistente(
        nombre=nombre,
        descripcion=descripcion,
        input_recomendado=input_recomendado,
        output_esperado=output_esperado,
        tecnologias=tecnologias,
        categoria=categoria,
        url_repo_pruebas=url_repo_pruebas,
        fecha_prueba=fecha_prueba,
        estado_prueba=estado_prueba,
        comentarios=comentarios
    )
    session.add(nuevo_asistente)
    session.commit()
    session.refresh(nuevo_asistente)

    # Agregar al vectorstore
    vectorstore = get_vectorstore()
    texto_final = asistente_to_text(nuevo_asistente)
    vectorstore.add_texts(
        texts=[texto_final],
        ids=[str(nuevo_asistente.id)],
        metadatas=[{"id": nuevo_asistente.id}]
    )
    # No es necesario llamar a persist()

    print(f"Asistente '{nombre}' agregado a Postgres y Chroma.")

# --------------------------
# CRUD: Actualizar asistente
# --------------------------
def update_asistente(
    asistente_id: int,
    nombre: str = None,
    descripcion: str = None,
    input_recomendado: str = None,
    output_esperado: str = None,
    tecnologias: List[str] = None,
    categoria: str = None,
    url_repo_pruebas: str = None,
    estado_pruebas: str = None,
    comentarios: str = None
):
    a = session.query(Asistente).filter_by(id=asistente_id).first()
    if not a:
        print(f"No existe un asistente con id {asistente_id}")
        return

    # Actualizar campos solo si no son None
    if nombre is not None:
        a.nombre = nombre
    if descripcion is not None:
        a.descripcion = descripcion
    if input_recomendado is not None:
        a.input_recomendado = input_recomendado
    if output_esperado is not None:
        a.output_esperado = output_esperado
    if tecnologias is not None:
        a.tecnologias = tecnologias
    if categoria is not None:
        a.categoria = categoria
    if url_repo_pruebas is not None:
        a.url_repo_pruebas = url_repo_pruebas
    if estado_pruebas is not None:
        a.estado_pruebas = estado_pruebas
    if comentarios is not None:
        a.comentarios = comentarios

    session.commit()
    session.refresh(a)

    # Actualizar en vectorstore: primero borrar el existente
    vectorstore = get_vectorstore()
    vectorstore.delete(ids=[str(a.id)])
    # Añadir el nuevo embedding
    texto_final = asistente_to_text(a)
    vectorstore.add_texts(
        texts=[texto_final],
        ids=[str(a.id)],
        metadatas=[{"id": a.id}]
    )
    # No es necesario llamar a persist()

    print(f"Asistente con id {asistente_id} actualizado en Postgres y Chroma.")

# --------------------------
# CRUD: Eliminar asistente
# --------------------------
def delete_asistente(asistente_id: int):
    a = session.query(Asistente).filter_by(id=asistente_id).first()
    if not a:
        print(f"No existe un asistente con id {asistente_id}")
        return

    session.delete(a)
    session.commit()

    # Eliminar del vectorstore
    vectorstore = get_vectorstore()
    vectorstore.delete(ids=[str(asistente_id)])
    # No es necesario llamar a persist()

    print(f"Asistente con id {asistente_id} eliminado de Postgres y Chroma.")

# --------------------------
# Reconstruir el vectorstore completo desde la BD
# --------------------------
def build_vectorstore_from_db():
    asistentes_db = session.query(Asistente).all()
    if not asistentes_db:
        print("No hay asistentes en Postgres. El vectorstore quedará vacío.")
        return None

    texts = []
    metadatas = []
    ids = []
    for a in asistentes_db:
        t = asistente_to_text(a)
        texts.append(t)
        metadatas.append({"id": a.id})
        ids.append(str(a.id))

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=lc_embeddings,
        metadatas=metadatas,
        ids=ids,
        collection_name="asistentes_collection",
        persist_directory="chroma_db"
    )
    # No es necesario llamar a persist()
    return vectorstore

#vectorstore = build_vectorstore_from_db()
vectorstore = get_vectorstore()
retriever = None
if vectorstore:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --------------------------
# Configurar Watsonx Granite LLM **(FUNCIONA MUCHO MEJOR CON claude-3.5-haiku DE ANTHROPIC, SONET SALE CARO)**
# --------------------------
#WATSONX_URL = os.environ["WATSONX_URL"]
#WATSONX_APIKEY = os.environ["WATSONX_APIKEY"]
#WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]

#credentials = Credentials(
#    url=WATSONX_URL,
#    api_key=WATSONX_APIKEY,
#)
#api_client = APIClient(credentials=credentials, project_id=WATSONX_PROJECT_ID)

#parameters = {
#   GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
#    GenParams.MIN_NEW_TOKENS: 1,
#    GenParams.MAX_NEW_TOKENS: 500,
#    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
#}

#model_id = ModelTypes.GRANITE_13B_INSTRUCT_V2
#watsonx_granite = WatsonxLLM(
#    model_id=model_id.value,
#    url=credentials.get("url"),
#    apikey=credentials.get("apikey"),
#    project_id=WATSONX_PROJECT_ID,
#    params=parameters
#)

# Inicializar LLM Anthropic Claude 3.5 Haiku
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

from langchain.prompts import PromptTemplate

prompt_template = """
    Tienes acceso a los siguientes asistentes de IA:

    {context}

    ---

    **Ejemplo de cómo responder:**

    **Consulta:**
    Necesito un asistente de IA para pruebas automatizadas en JIRA.

    **Respuesta:**
    Te recomiendo usar AgileQA. Este asistente integra pruebas automatizadas con JIRA, lo que facilita el seguimiento ágil de tus proyectos. Puede ayudarte a generar reportes automáticos sobre el estado de las pruebas y asegurar que se alineen con los tickets de JIRA, mejorando la eficiencia y la coordinación en tu equipo.

    ---

    **Tu tarea:**
    Selecciona el asistente de IA más adecuado para la siguiente consulta y explica detalladamente por qué es la mejor opción. Proporciona el nombre del asistente y una justificación clara basada en las descripciones proporcionadas.
    En caso de que no haya ningún asistente relevante en el contexto dímelo.

    **Consulta:**
    {question}

    **Respuesta (nombre del asistente y justificación):**
    """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

keyword_prompt = PromptTemplate(
    input_variables=["query"],
    template="Extrae las 4 palabras clave más importantes de la consulta '{query}' y sepáralas por comas."
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


def search_db_by_keywords(keywords: str):
    keywords = keywords.strip()
    if "," in keywords:
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    else:
        # Si no hay comas, dividimos por espacio
        keywords_list = keywords.split()

    conditions = []
    for kw in keywords_list:
        conditions.append(
            f"(array_to_string(keywords, ' ') ILIKE '%{kw}%' OR nombre ILIKE '%{kw}%' OR descripcion ILIKE '%{kw}%')"
        )

    if not conditions:
        # Si no hay conditions, devolver vacío para evitar query inválida
        return []

    where_clause = " OR ".join(conditions)
    query = text(f"SELECT * FROM asistentes WHERE {where_clause} LIMIT 8")

    with engine.connect() as conn:
        result = conn.execute(query)
        results = [dict(r._mapping) for r in result]
    return results



search_db_tool = Tool(
    name="search_db",
    func=search_db_by_keywords,
    description="Usar esta herramienta para buscar asistentes en la base de datos relacional usando keywords extraidas."
)

def search_vectorstore(query: str):
    docs = vectorstore.similarity_search(query, k=5)
    return [d.page_content for d in docs]

vectorstore_tool = Tool(
    name="search_vectorstore",
    func=search_vectorstore,
    description="Usar esta herramienta para buscar asistentes en la base vectorial dado un query."
)



agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
        Eres un agente que ayuda a seleccionar el mejor asistente de IA. La secuencia recomendada:

        1. Extrae las palabras clave de la consulta del usuario (ya se hizo y se pasará en el input).
        2. Usa 'search_db' con esas palabras clave para ver si hay asistentes relevantes en la base relacional.
        3. Usa 'search_vectorstore' con la consulta original para obtener asistentes vectoriales relevantes.
        4. Combina los resultados y da la mejor recomendación.

        Consulta del usuario: {input}

        {agent_scratchpad}
        """)

tools = [search_db_tool, vectorstore_tool]

agent = initialize_agent( tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prompt=agent_prompt )






app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "http://localhost:4200"}})

""" @app.route("/query", methods=["POST"])
def get_response():
    if not retriever:
        return jsonify({"error": "No vectorstore found. Please load assistants first."}), 500

    data = request.get_json()
    query = data.get("query", "")
    print(f"Consulta recibida: {query}")

    relevant_docs = retriever.get_relevant_documents(query)
    print(f"Documentos recuperados para la consulta '{query}': {len(relevant_docs)}")

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("Contexto enviado al LLM:")
    print(context if context.strip() else "Contexto vacío.")

    if not context.strip():
        print("No se encontraron documentos relevantes para la consulta.")
        return jsonify({"query": query, "response": "No se encontraron asistentes relevantes para tu consulta."})

    full_prompt = prompt.format(context=context, question=query)
    print("Prompt final enviado al LLM:")
    print(full_prompt)

    #response = watsonx_granite(full_prompt)
    response = llm(full_prompt)
    print("Respuesta del LLM:")
    print(response)

    #return jsonify({"query": query, "response": response})
    return jsonify({"query": query, "response": response.content})
 """
@app.route("/query", methods=["POST"])
def get_response():
    data = request.get_json()
    query = data.get("query", "")
    print(f"Consulta recibida: {query}")

    # 1. Generar las keywords
    keywords = keyword_chain.run(query=query)
    print("Keywords generadas:", keywords)

    # 2. Ejecutar el agente
    # Aquí le pasamos la consulta original y las keywords en el input del agente
    # para que él sepa qué hacer. Por ejemplo:
    agente_input = f"La consulta del usuario es: '{query}'. Las keywords son: {keywords}"
    final_response = agent.run(agente_input)

    return jsonify({"query": query, "response": final_response})

if __name__ == "__main__":
    # Ejemplo de uso:
    #load_assistants_from_excel("asistentes_desarrollo.xlsx")
    # add_asistente_manualmente(...)
    # update_asistente(asistente_id=1, nombre="Nuevo Nombre")
    # delete_asistente(asistente_id=1)
    # build_vectorstore_from_db()  # Para regenerar vectorstore completo
    app.run(host="0.0.0.0", port=8000, debug=True)

