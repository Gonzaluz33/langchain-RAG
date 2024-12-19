import ollama
from typing import List
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from database import session, Asistente



# --------------------------
# Embeddings con Ollama
# --------------------------------
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
    tech_list = asistente.tecnologias if asistente.tecnologias else []
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
    return vectorstore