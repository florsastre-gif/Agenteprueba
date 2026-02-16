import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. CARGA DE CONFIGURACIÓN
load_dotenv()

# 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS)
# Configuración robusta para evitar el error de ImportError en DuckDuckGo
wrapper = DuckDuckGoSearchAPIWrapper(region="es-es", time="y", max_results=2)
search = DuckDuckGoSearchRun(api_wrapper=wrapper)

# Configuración de Wikipedia en español
wiki_wrapper = WikipediaAPIWrapper(lang="es", top_k_results=1)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_wrapper)

@tool
def calculadora_iva(precio_neto: float, tasa: float = 21) -> str:
    """Calcula el precio final con IVA y el monto del impuesto."""
    impuesto = precio_neto * (tasa / 100)
    total = precio_neto + impuesto
    return f"Precio neto: {precio_neto}€, IVA ({tasa}%): {impuesto}€, Total: {total}€"

tools = [search, wikipedia, calculadora_iva]

# 3. CONFIGURACIÓN DEL MODELO Y PROMPT
# El modelo busca automáticamente GOOGLE_API_KEY en los Secrets de Streamlit
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
st.markdown("Busco información en tiempo real, Wikipedia y calculo impuestos.")

user_input = st.text_input("Haz una consulta al agente:", placeholder="Ej: ¿Cuánto cuesta una suscripción a Netflix y cuánto sería con 21% de IVA?")

if user_input:
    with st.spinner("El agente está trabajando..."):
        try:
            # Ejecución del agente
            resultado = agent_executor.invoke({"input": user_input})
            
            # Mostrar respuesta final en la web
            st.subheader("Respuesta:")
            st.write(resultado["output"])
            
        except Exception as e:
            st.error(f"Hubo un error. Verifica que tu API KEY esté en los Secrets de Streamlit.")
            st.exception(e)

if __name__ == "__main__":
    pass
