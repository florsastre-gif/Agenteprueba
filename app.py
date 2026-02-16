import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. CARGA DE CONFIGURACIÓN
# Intenta cargar desde .env (local) o desde los Secrets (Streamlit Cloud)
load_dotenv()

# 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS)
# Inicialización estable de buscadores
search = DuckDuckGoSearchRun()
api_wrapper = WikipediaAPIWrapper(lang="es", top_k_results=1)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)

@tool
def calculadora_iva(precio_neto: float, tasa: float = 21) -> str:
    """Calcula el precio final con IVA y el monto del impuesto."""
    impuesto = precio_neto * (tasa / 100)
    total = precio_neto + impuesto
    return f"Precio neto: {precio_neto}€, IVA ({tasa}%): {impuesto}€, Total: {total}€"

# Lista de herramientas que el agente puede usar
tools = [search, wikipedia, calculadora_iva]

# 3. CONFIGURACIÓN DEL MODELO Y PROMPT
# El modelo leerá automáticamente la clave GOOGLE_API_KEY del entorno
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto que usa búsqueda web y Wikipedia para dar datos precisos y actuales."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. CONSTRUCCIÓN DEL AGENTE
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# 5. INTERFAZ DE STREAMLIT
st.set_page_config(page_title="Mi Agente IA", page_icon="🤖")
st.title("🤖 Mi Agente con LangChain")
st.markdown("Busco información en tiempo real, consulto Wikipedia y calculo impuestos.")

# Input del usuario
user_input = st.text_input("Haz una consulta al agente:", placeholder="Ej: ¿Cuál es el precio de la PS5 y cuánto sería con 21% de IVA?")

if user_input:
    with st.spinner("El agente está razonando y buscando información..."):
        try:
            # Ejecución del agente
            resultado = agent_executor.invoke({"input": user_input})
            
            # Mostrar respuesta final en la interfaz web
            st.subheader("Respuesta del Agente:")
            st.write(resultado["output"])
            
        except Exception as e:
            st.error("Error de configuración o conexión.")
            st.write("Asegúrate de haber configurado tu GOOGLE_API_KEY en los Secrets de Streamlit.")
            st.exception(e)

# Ejecución local (Opcional)
if __name__ == "__main__":
    pass
