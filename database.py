import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, Date, Float, ForeignKey, ARRAY, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from langchain import SQLDatabase
from datetime import date

load_dotenv()

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


def add_asistente_manualmente(
    nombre: str,
    descripcion: str,
    input_recomendado: str,
    output_esperado: str,
    tecnologias: list,
    categoria: str,
    url_repo_pruebas: str = None,
    fecha_prueba: date = None,
    estado_prueba: str = None,
    comentarios: str = None
):
    existing_asistente = session.query(Asistente).filter_by(nombre=nombre).first()
    if existing_asistente:
        print(f"Ya existe un asistente con el nombre '{nombre}'.")
        return None

    nuevo_asistente = Asistente(
        nombre=nombre,
        descripcion=descripcion,
        input_recomendado=input_recomendado,
        output_esperado=output_esperado,
        tecnologias=tecnologias,
        categoria=categoria,
        url_repo_pruebas=url_repo_pruebas,
        estado_pruebas=estado_prueba,
        comentarios=comentarios
    )
    session.add(nuevo_asistente)
    session.commit()
    session.refresh(nuevo_asistente)
    return nuevo_asistente

def update_asistente(asistente_id: int, **kwargs):
    a = session.query(Asistente).filter_by(id=asistente_id).first()
    if not a:
        print(f"No existe un asistente con id {asistente_id}")
        return None

    for key, value in kwargs.items():
        if value is not None and hasattr(a, key):
            setattr(a, key, value)

    session.commit()
    session.refresh(a)
    return a

def delete_asistente(asistente_id: int):
    a = session.query(Asistente).filter_by(id=asistente_id).first()
    if not a:
        print(f"No existe un asistente con id {asistente_id}")
        return False

    session.delete(a)
    session.commit()
    return True

def get_all_asistentes():
    return session.query(Asistente).all()
