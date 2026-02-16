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
load_dotenv()

# 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS)
search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def calculadora_iva(precio_neto: float, tasa: float = 21) -> str:
    """Calcula el precio final con IVA y el monto del impuesto."""
    impuesto = precio_neto * (tasa / 100)
    total = precio_neto + impuesto
    return f"Precio neto: {precio_neto}€, IVA ({tasa}%): {impuesto}€, Total: {total}€"

tools = [search, wikipedia, calculadora_iva]

# 3. CONFIGURACIÓN DEL MODELO Y PROMPT
# Asegúrate de tener la clave en Secrets de Streamlit o archivo .env
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente experto que usa búsqueda web y Wikipedia para dar datos precisos."),
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
st.markdown("Busco información en tiempo real y calculo impuestos.")

user_input = st.text_input("Escribe tu consulta aquí:", placeholder="Ej: ¿Cuál es el precio del iPhone 15 y cuánto sería con 21% de IVA?")

if user_input:
    with st.spinner("El agente está trabajando..."):
        try:
            # Ejecución del agente
            resultado = agent_executor.invoke({"input": user_input})
            
            # Mostrar respuesta final en la web
            st.subheader("Respuesta:")
            st.write(resultado["output"])
            
        except Exception as e:
            st.error(f"Hubo un error de configuración. Verifica tu API KEY en los Secrets de Streamlit.")
            st.exception(e)

if __name__ == "__main__":
    # Para ejecutar en local usa: streamlit run app.py
    pass
