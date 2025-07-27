#Realizado por José de Jesús Duana Ramos
import os
import streamlit as st
import re #Esto lo utilizamos para que pueda dar respuesta con imágenes desde los markdown
import pandas as pd #Estas dos las vamos a utilizar para el dashboard
import datetime
from dotenv import load_dotenv
from pathlib import Path

# LangChain y OpenAI
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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
def preparar_asistente():
    st.info("🔄 Cargando e indexando documentos...")
    docs = cargar_documentos_desde_carpeta("knowledge")

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=50)#se agrega el separator="\n\n" para darle más contexto completo al modelo para que vea la tabla como un bloque
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
# Prompt personalizado
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
    
    # 🧠 Historial de conversación
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4.1-nano", temperature=0),
        retriever=vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # Indica en dónde se debe guardar la respuesta
    )

    print("Retornando:", qa, memory, len(docs))
    return qa, memory, len(docs)


# Inicializar asistente
qa, memory, total_docs = preparar_asistente()
st.success(f"✅ Documentos cargados: {total_docs}")

# Inicializar historial
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


#Parte que guarda en la memoria
def guardar_historial_de_memory(memory, archivo="historial.csv"):
    mensajes = memory.chat_memory.messages
    pares = []

    # Leer historial actual si existe
    historial_existente = set()
    if os.path.exists(archivo):
        df_existente = pd.read_csv(archivo)
        historial_existente = set(zip(df_existente["pregunta"], df_existente["respuesta"]))

    # Generar pares nuevos y comparar
    for i in range(0, len(mensajes), 2):
        if i + 1 < len(mensajes):
            pregunta = mensajes[i].content
            respuesta = mensajes[i + 1].content

            if (pregunta, respuesta) not in historial_existente:
                pares.append({
                    "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pregunta": pregunta,
                    "respuesta": respuesta
                })

    # Guardar solo si hay nuevos pares
    if pares:
        df_nuevos = pd.DataFrame(pares)
        df_nuevos.to_csv(archivo, mode='a', header=not os.path.exists(archivo), index=False, encoding='utf-8-sig')


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
        ruta_completa = os.path.join("knowledge", ruta)
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



with st.sidebar: #Todo este apartado crea la vista de admin en streamlit
    with st.expander("🔐 Mostrar acceso administrativo"):
        st.markdown("## Acceso administrativo")
        clave_admin = st.text_input("Contraseña", type="password")

        if clave_admin == "admin123":  # ✅ Cambia la clave si lo deseas
            st.success("Acceso concedido")

            if os.path.exists("historial.csv"):
                st.markdown("### 🧠 Historial de preguntas y respuestas")
                df_hist = pd.read_csv("historial.csv")

                st.dataframe(df_hist, use_container_width=True)

                st.download_button(
                    label="⬇️ Descargar historial como CSV",
                    data=df_hist.to_csv(index=False).encode("utf-8-sig"),
                    file_name="historial.csv",
                    mime="text/csv"
                )
            else:
                st.info("No hay historial registrado aún.")
        elif clave_admin:
            st.error("Contraseña incorrecta.")

