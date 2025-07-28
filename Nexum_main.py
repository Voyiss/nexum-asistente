#Realizado por José de Jesús Duana Ramos
import os
import streamlit as st
import re #Esto lo utilizamos para que pueda dar respuesta con imágenes desde los markdown
import pandas as pd #Estas dos las vamos a utilizar para el dashboard
import datetime
from dotenv import load_dotenv
from pathlib import Path
import yaml
import sqlite3

# LangChain y OpenAI
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory #Se usará para guardar la memoria del historial de conversaciones
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    CSVLoader
)

# Cargar configuración desde el archivo YAML
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# --- NUEVA FUNCIÓN PARA INICIALIZAR LA BASE DE DATOS ---
def inicializar_db():
    """Crea la tabla de historial en la DB si no existe."""
    db_path = config['paths']['history_file']
    # Crear la carpeta 'data' si no existe
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            pregunta TEXT NOT NULL,
            respuesta TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# --- LLAMAR A LA FUNCIÓN AL INICIO DEL SCRIPT ---
inicializar_db()

# Cargar la API Key desde .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración de Streamlit
st.set_page_config(page_title="Nexum Assistant", page_icon="🧠")
st.title("🧠 Asistente NEXUM - DEMO")

# 🧠 Función para cargar todos los documentos desde knowledge/
def cargar_documentos_desde_carpeta(carpeta_base):
    documentos = []
    ruta = Path(carpeta_base)

    for archivo in ruta.rglob("*"):  # Recorre subcarpetas
        try:
            if archivo.suffix in [".txt", ".md"]:
                loader = TextLoader(str(archivo), encoding='utf-8') #Es importante el utf-8, ya que sin esto puede marcar un error al momento de leer los archivos.
            elif archivo.suffix == ".pdf":
                loader = PyPDFLoader(str(archivo), encoding='utf-8')
            elif archivo.suffix == ".docx":
                loader = UnstructuredWordDocumentLoader(str(archivo), encoding='utf-8')
            elif archivo.suffix == ".xlsx":
                loader = UnstructuredExcelLoader(str(archivo), encoding='utf-8')
            elif archivo.suffix == ".csv":
                loader = CSVLoader(str(archivo))
            else:
                continue  # Ignorar archivos no compatibles

            documentos += loader.load()
        except Exception as e:
            st.warning(f"Error al cargar {archivo.name}: {e}")

    return documentos

# ⚙️ Preparar asistente
@st.cache_resource
# REEMPLAZA TU FUNCIÓN ANTIGUA CON ESTA VERSIÓN MEJORADA
@st.cache_resource
def preparar_asistente():
    """
    Prepara el asistente. Si la base de datos de vectores ya existe, la carga.
    Si no, procesa los documentos y la crea.
    """
    vector_db_path = config['paths']['vector_store']
    
    # Prepara el almacén de vectores con ChromaDB
    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=OpenAIEmbeddings()
    )

    # Comprueba si la base de datos ya tiene documentos
    docs_existentes = vectorstore.get().get('ids', [])

    if not docs_existentes:
        st.info("🔄 Es la primera vez que se ejecuta. Indexando documentos...")
        # Carga los documentos
        docs = cargar_documentos_desde_carpeta(config['paths']['knowledge_base'])
        if not docs:
            st.error("No se encontraron documentos en la carpeta 'knowledge'. El asistente no puede funcionar sin datos.")
            return None, None
        
        # Divide los documentos
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # Añade los documentos a ChromaDB. Esto los guarda en el disco.
        vectorstore.add_documents(documents=splits, embedding=OpenAIEmbeddings())
        st.success(f"✅ Documentos indexados y guardados permanentemente.")
    else:
        st.success(f"✅ Base de conocimiento cargada desde el disco.")


    # Prompt personalizado (sin cambios)
    prompt = PromptTemplate.from_template("""
Eres un asistente virtual llamado Nexum. Tu objetivo es ayudar a los empleados a entender temas internos de la empresa de forma clara, precisa y profesional.

Tu estilo es:
- Formal pero cercano
- Siempre respetuoso
- Evita tecnicismos innecesarios
- Si no sabes algo, lo reconoces y propones buscar una respuesta

Cuando respondas, **si hay tablas, en los documentos, respétalos y muéstralos sin modificar**.
                                          
Basado en el siguiente contexto:

{context}

Pregunta del usuario: {question}
""")
    
    # Historial de conversación (sin cambios)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Creación del chain (sin cambios)
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature']
        ),
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    
    return qa, memory


# Inicializar asistente
qa, memory = preparar_asistente()
# El mensaje de éxito ahora se muestra dentro de la propia función.

# Inicializar historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#Parte que guarda en la memoria
# REEMPLAZA TU FUNCIÓN ANTIGUA CON ESTA
def guardar_historial_de_memory(memory):
    """Guarda los nuevos pares de pregunta/respuesta en la base de datos SQLite."""
    db_path = config['paths']['history_file']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Obtener el último par de mensajes (pregunta del usuario y respuesta del AI)
    mensajes = memory.chat_memory.messages
    if len(mensajes) >= 2:
        pregunta = mensajes[-2].content
        respuesta = mensajes[-1].content
        fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insertar el nuevo par en la base de datos
        cursor.execute(
            "INSERT INTO historial (fecha, pregunta, respuesta) VALUES (?, ?, ?)",
            (fecha, pregunta, respuesta)
        )
        conn.commit()
    
    conn.close()


# Interfaz de usuario
pregunta = st.text_input("Haz tu pregunta:", placeholder="Ej: ¿Dónde descargo mi recibo de nómina?")
chat_history = []  # Puedes hacerlo persistente si quieres un historial más largo

def mostrar_respuesta_con_imagenes(texto): #Esta parte del código permite leer las imágenes que se incrustan en la base de conocimientos en Markdown.
    #Recuerda que se debe poner la sintaxis ![Título opcional](assets/nombrearchivo.png) en donde "Titulo opcional" es el título cualquiera,
    # "assets" es al carpeta en knowledge y "nombrearchivo" es el nombre del file con su extensión.
    # Buscar imágenes en formato markdown ![alt](ruta)
    patron = r'!\[.*?\]\((.*?)\)'
    rutas_imagenes = re.findall(patron, texto)

    # Mostrar texto sin las etiquetas de imagen
    texto_sin_imagenes = re.sub(patron, '', texto)
    st.markdown(texto_sin_imagenes)

    # Mostrar cada imagen si existe
    for ruta in rutas_imagenes:
        ruta_completa = os.path.join(config['paths']['knowledge_base'], ruta)
        if os.path.exists(ruta_completa):
            st.image(ruta_completa, use_container_width=True)
        else:
            st.warning(f"⚠️ Imagen no encontrada: {ruta_completa}")


if pregunta:
    with st.spinner("Pensando..."):
        respuesta = qa({
            "question": pregunta,
            "chat_history": st.session_state.chat_history
        })
        st.markdown("### 🧠 Respuesta:")
        mostrar_respuesta_con_imagenes(respuesta['answer']) #antes de utilizar imágenes, solo utilizaba el: st.write(respuesta['answer'])

        # Guardar historial de chat
        guardar_historial_de_memory(memory)



# Quitamos la contraseña de aquí
load_dotenv() # Asegúrate que load_dotenv() esté al inicio del script
ADMIN_PASSWORD_ENV = os.getenv("ADMIN_PASSWORD")

with st.sidebar:
    with st.expander("🔐 Mostrar acceso administrativo"):
        st.markdown("## Acceso administrativo")
        clave_admin = st.text_input("Contraseña", type="password")

        if clave_admin and clave_admin == ADMIN_PASSWORD_ENV: # ✅ MÁS SEGURO
            st.success("Acceso concedido")
            # ... (el resto de tu código para mostrar el historial)
        elif clave_admin:
            st.error("Contraseña incorrecta.")

            if os.path.exists(db_path):
                st.markdown("### 🧠 Historial de preguntas y respuestas")
                
                # Conectar a la base de datos y cargar en un DataFrame
                conn = sqlite3.connect(db_path)
                df_hist = pd.read_sql_query("SELECT fecha, pregunta, respuesta FROM historial ORDER BY fecha DESC", conn)
                conn.close()

                st.dataframe(df_hist, use_container_width=True)

                st.download_button(
                    label="⬇️ Descargar historial como CSV",
                    data=df_hist.to_csv(index=False).encode("utf-8-sig"),
                    file_name="historial_exportado.csv", # Le ponemos un nombre diferente para la exportación
                    mime="text/csv"
                )
            else:
                st.info("No hay historial registrado aún.")
        elif clave_admin:
            st.error("Contraseña incorrecta.")

