import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Float, ForeignKey, ARRAY, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from langchain import SQLDatabase
from datetime import date
from typing import List
import pandas as pd

load_dotenv()


# --------------------------
# Configuración de la BD Postgres (FUENTE DE VERDAD)
# --------------------------------------------
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
    metrica_eficiencia = Column(Float)
    metrica_precision = Column(Float)
    resumen = Column(Text)
    conclusion = Column(Text)
    fecha_ejecucion = Column(Date)

class Categoria(Base):
    __tablename__ = 'categorias'
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False)

engine = create_engine(DATABASE_URL, echo=False)
db = SQLDatabase.from_uri(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)

def search_db_by_keywords(keywords: str):
    keywords = keywords.strip()
    if "," in keywords:
        keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    else:
        keywords_list = keywords.split()

    conditions = []
    for kw in keywords_list:
        conditions.append(
            f"(array_to_string(keywords, ' ') ILIKE '%{kw}%' OR nombre ILIKE '%{kw}%' OR descripcion ILIKE '%{kw}%')"
        )

    if not conditions:
        return []

    where_clause = " OR ".join(conditions)
    query = text(f"SELECT * FROM asistentes WHERE {where_clause} LIMIT 8")

    with engine.connect() as conn:
        result = conn.execute(query)
        results = [dict(r._mapping) for r in result]
    return results


from vectorstore import get_vectorstore, asistente_to_text


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

    vectorstore = get_vectorstore()
    texto_final = asistente_to_text(nuevo_asistente)
    vectorstore.add_texts(
        texts=[texto_final],
        ids=[str(nuevo_asistente.id)],
        metadatas=[{"id": nuevo_asistente.id}]
    )

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

    print(f"Asistente con id {asistente_id} eliminado de Postgres y Chroma.")


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

    vectorstore = get_vectorstore()
    for a, t in zip(nuevos_asistentes, assistants_data):
        vectorstore.add_texts(
            texts=[t],
            ids=[str(a.id)],
            metadatas=[{"id": a.id}]
        )
    print("Asistentes agregados a Chroma.")