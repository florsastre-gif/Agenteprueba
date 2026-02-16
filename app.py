import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. CARGA DE CONFIGURACIÓN SEGURA
# Carga la API KEY desde tu archivo .env (que Git ignorará)
load_dotenv() 

# 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS)
search = DuckDuckGoSearchRun()

@tool
def calculadora_iva(precio_neto: float, tasa: float = 21) -> str:
    """Calcula el precio final con IVA y el monto del impuesto."""
    impuesto = precio_neto * (tasa / 100)
    total = precio_neto + impuesto
    return f"Precio neto: {precio_neto}€, IVA ({tasa}%): {impuesto}€, Total: {total}€"

# Lista de herramientas disponibles para el agente
tools = [search, calculadora_iva]

# 3. CONFIGURACIÓN DEL MODELO Y PROMPT
# Usamos gemini-1.5-flash por estabilidad y rapidez
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente de compras experto. Buscas precios actuales y calculas impuestos."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. CONSTRUCCIÓN DEL AGENTE (Basado en el análisis de tu notebook)
# El agente decide qué herramienta usar basándose en la descripción de las mismas
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, # Muestra el razonamiento del agente en la terminal
    handle_parsing_errors=True
)

# 5. EJECUCIÓN DEL SCRIPT
if __name__ == "__main__":
    query = "¿Cuál es el precio del abono transporte en Madrid en 2026? Dime el precio con un IVA aplicado del 10% si el que encuentras no lo tiene."
    resultado = agent_executor.invoke({"input": query})
    print("\n--- RESPUESTA FINAL ---")
    print(resultado["output"])