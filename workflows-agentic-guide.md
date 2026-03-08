# Guía de Implementación: Arquitectura de Sistemas Agénticos en Python

> **Basada en el marco estratégico de integración de inteligencia autónoma para equipos de desarrollo de software.**

---

## Tabla de Contenidos

1. [Requisitos y Setup del Entorno](#1-requisitos-y-setup-del-entorno)
2. [Estructura del Proyecto](#2-estructura-del-proyecto)
3. [Componentes Fundamentales del Agente](#3-componentes-fundamentales-del-agente)
4. [Implementación de Agentes Especializados](#4-implementación-de-agentes-especializados)
5. [Patrones de Orquestación](#5-patrones-de-orquestación)
6. [Gestión de Memoria y Bases de Datos Vectoriales](#6-gestión-de-memoria-y-bases-de-datos-vectoriales)
7. [Model Context Protocol (MCP) y Herramientas](#7-model-context-protocol-mcp-y-herramientas)
8. [Observabilidad y Trazado](#8-observabilidad-y-trazado)
9. [Ciclo de Vida del Desarrollo Agéntico (ADLC)](#9-ciclo-de-vida-del-desarrollo-agéntico-adlc)
10. [Estrategias de Manejo de Errores](#10-estrategias-de-manejo-de-errores)
11. [Testing y Evaluación](#11-testing-y-evaluación)
12. [KPIs y Medición de ROI](#12-kpis-y-medición-de-roi)

---

## 1. Requisitos y Setup del Entorno

### Dependencias principales

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install \
  langchain==0.2.0 \
  langchain-openai==0.1.0 \
  langgraph==0.1.0 \
  langsmith==0.1.0 \
  chromadb==0.5.0 \
  openai==1.30.0 \
  anthropic==0.28.0 \
  pydantic==2.7.0 \
  python-dotenv==1.0.0 \
  fastapi==0.111.0 \
  pytest==8.2.0 \
  pytest-asyncio==0.23.0
```

### Variables de entorno

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-dev-team
CHROMA_PERSIST_DIR=./data/chroma
```

---

## 2. Estructura del Proyecto

```
agentic_team/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Clase base con componentes comunes
│   ├── architect_agent.py     # Agente Arquitecto
│   ├── developer_agent.py     # Agente Desarrollador
│   ├── reviewer_agent.py      # Agente Revisor
│   └── pm_agent.py            # Agente Product Manager
├── orchestration/
│   ├── __init__.py
│   ├── sequential.py          # Orquestación secuencial
│   ├── concurrent.py          # Orquestación concurrente
│   └── group_chat.py          # Orquestación de chat grupal
├── memory/
│   ├── __init__.py
│   ├── short_term.py          # Memoria a corto plazo
│   └── long_term.py           # Memoria a largo plazo (vectorial)
├── tools/
│   ├── __init__.py
│   ├── git_tools.py           # Herramientas de control de versiones
│   ├── code_tools.py          # Herramientas de análisis de código
│   └── api_tools.py           # Herramientas de integración externa
├── observability/
│   ├── __init__.py
│   ├── tracer.py              # Trazado distribuido
│   └── metrics.py             # Métricas y KPIs
├── governance/
│   ├── __init__.py
│   ├── prompt_registry.py     # Registro centralizado de prompts
│   └── version_control.py     # Control de versiones de agentes
├── evaluation/
│   ├── __init__.py
│   ├── golden_dataset.py      # Conjuntos dorados de prueba
│   └── llm_judge.py           # Evaluador LLM-as-a-judge
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── config.py
└── main.py
```

---

## 3. Componentes Fundamentales del Agente

### Clase base del agente

```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import uuid
import logging

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Estado interno del agente."""
    agent_id: str
    task: str
    context: Dict[str, Any] = {}
    messages: List[Dict] = []
    tool_calls: List[Dict] = []
    feedback_loop: List[Dict] = []
    status: str = "idle"  # idle, running, completed, failed


class BaseAgent(ABC):
    """
    Clase base para todos los agentes del equipo.
    Implementa los 4 componentes: percepción, razonamiento, acción y feedback.
    """

    def __init__(
        self,
        name: str,
        role: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        memory=None,
        tools: Optional[List] = None,
    ):
        self.name = name
        self.role = role
        self.agent_id = str(uuid.uuid4())
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.memory = memory
        self.tools = tools or []
        self.state = AgentState(agent_id=self.agent_id, task="")

        # Vincular herramientas al LLM si existen
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    # ─── Percepción ─────────────────────────────────────────────────────────────
    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa y estructura la información de entrada."""
        self.state.context.update(input_data)
        logger.info(f"[{self.name}] Percibiendo entrada: {list(input_data.keys())}")
        return self.state.context

    # ─── Razonamiento ────────────────────────────────────────────────────────────
    @abstractmethod
    def reason(self, task: str, context: Dict[str, Any]) -> str:
        """Define el razonamiento específico de cada agente."""
        pass

    # ─── Acción ─────────────────────────────────────────────────────────────────
    def act(self, reasoning_output: str) -> Dict[str, Any]:
        """Ejecuta acciones basadas en el razonamiento."""
        result = {"output": reasoning_output, "agent": self.name, "status": "completed"}
        self.state.tool_calls.append(result)
        return result

    # ─── Bucle de Retroalimentación ──────────────────────────────────────────────
    def feedback(self, action_result: Dict[str, Any]) -> None:
        """Actualiza el estado interno basándose en el resultado de la acción."""
        self.state.feedback_loop.append(action_result)
        self.state.status = action_result.get("status", "completed")
        logger.info(f"[{self.name}] Feedback recibido: {action_result['status']}")

    # ─── Ciclo completo ──────────────────────────────────────────────────────────
    def run(self, task: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Ejecuta el ciclo completo: percepción → razonamiento → acción → feedback."""
        self.state.task = task
        self.state.status = "running"

        enriched_context = self.perceive(context)
        reasoning = self.reason(task, enriched_context)
        result = self.act(reasoning)
        self.feedback(result)

        return result
```

---

## 4. Implementación de Agentes Especializados

### Agente Arquitecto

```python
# agents/architect_agent.py
from agents.base_agent import BaseAgent
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


ARCHITECT_SYSTEM_PROMPT = """
Eres el Agente Arquitecto de un equipo de desarrollo de software.
Tu responsabilidad es:
- Leer requisitos y planificar el enfoque técnico
- Asignar tareas a otros agentes especializados
- NO escribir código directamente
- Garantizar que el sistema sea fiable y adaptativo a escala
- Anticipar modos de fallo y brechas de integración

Formato de salida: Siempre estructura tu respuesta como:
1. ANÁLISIS: Evaluación de los requisitos
2. PLAN: Lista de tareas ordenadas con responsable (agente)
3. RESTRICCIONES: Consideraciones de seguridad y rendimiento
4. CRITERIOS DE ÉXITO: Cómo se validará el resultado
"""


class ArchitectAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Arquitecto",
            role="Diseña la solución técnica y orquesta al equipo",
            **kwargs,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ARCHITECT_SYSTEM_PROMPT),
            ("human", "Tarea: {task}\nContexto: {context}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def reason(self, task: str, context: Dict[str, Any]) -> str:
        response = self.chain.invoke({"task": task, "context": str(context)})
        self.state.messages.append({"role": "architect", "content": response})
        return response
```

### Agente Desarrollador

```python
# agents/developer_agent.py
from agents.base_agent import BaseAgent
from tools.code_tools import run_code, lint_code, format_code
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DEVELOPER_SYSTEM_PROMPT = """
Eres el Agente Desarrollador de un equipo de software.
Tu responsabilidad es:
- Implementar código limpio basado en el plan del Arquitecto
- Ejecutar y validar que el código funciona antes de entregarlo
- Aplicar mejores prácticas: SOLID, DRY, KISS
- Documentar cada función con docstrings

Cuando generes código:
- Incluye siempre tests unitarios
- Maneja excepciones explícitamente
- Usa type hints en Python
"""


class DeveloperAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Desarrollador",
            role="Implementa el código según el diseño arquitectónico",
            tools=[run_code, lint_code, format_code],
            **kwargs,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DEVELOPER_SYSTEM_PROMPT),
            ("human", "Plan técnico: {task}\nContexto: {context}"),
        ])
        self.chain = self.prompt | self.llm_with_tools | StrOutputParser()

    def reason(self, task: str, context: Dict[str, Any]) -> str:
        response = self.chain.invoke({"task": task, "context": str(context)})
        return response
```

### Agente Revisor (patrón Evaluador-Optimizador)

```python
# agents/reviewer_agent.py
from agents.base_agent import BaseAgent
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json


REVIEWER_SYSTEM_PROMPT = """
Eres el Agente Revisor de código. Tu trabajo es evaluar el código generado.
Devuelve SIEMPRE un JSON con esta estructura exacta:
{
  "approved": true/false,
  "score": 0-100,
  "issues": ["issue1", "issue2"],
  "suggestions": ["sugerencia1", "sugerencia2"],
  "security_concerns": ["concern1"],
  "requires_retry": true/false
}
"""


class ReviewerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="Revisor",
            role="Evalúa y valida el código generado por el Desarrollador",
            **kwargs,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", "Código a revisar:\n{task}\n\nCriterios de evaluación: {context}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def reason(self, task: str, context: Dict[str, Any]) -> str:
        response = self.chain.invoke({"task": task, "context": str(context)})
        return response

    def evaluate(self, code: str, criteria: Dict) -> Dict[str, Any]:
        """Evalúa código y retorna un reporte estructurado."""
        raw = self.reason(code, criteria)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"approved": False, "score": 0, "issues": ["Parse error"], "requires_retry": True}
```

---

## 5. Patrones de Orquestación

### Orquestación Secuencial

```python
# orchestration/sequential.py
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class SequentialOrchestrator:
    """
    Encadena agentes en orden lineal.
    La salida de cada agente es la entrada del siguiente.
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run(self, initial_task: str, initial_context: Dict[str, Any] = {}) -> Dict[str, Any]:
        context = initial_context.copy()
        current_task = initial_task
        results = []

        for agent in self.agents:
            logger.info(f"Ejecutando agente: {agent.name}")
            result = agent.run(current_task, context)

            # La salida del agente enriquece el contexto para el siguiente
            context[f"{agent.name}_output"] = result["output"]
            current_task = result["output"]  # El output se convierte en la nueva tarea
            results.append(result)

        return {"results": results, "final_output": results[-1]["output"], "context": context}
```

### Orquestación Concurrente

```python
# orchestration/concurrent.py
import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class ConcurrentOrchestrator:
    """
    Ejecuta múltiples agentes en paralelo y agrega sus resultados.
    Ideal para lluvia de ideas y razonamiento de conjunto.
    """

    def __init__(self, agents: List[BaseAgent], aggregation: str = "vote"):
        self.agents = agents
        self.aggregation = aggregation  # "vote", "weighted", "llm_synthesis"

    async def _run_agent_async(self, agent: BaseAgent, task: str, context: Dict) -> Dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.run, task, context)

    async def run_async(self, task: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
        tasks = [self._run_agent_async(agent, task, context) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if not isinstance(r, Exception)]
        aggregated = self._aggregate(valid_results)

        return {"individual_results": valid_results, "aggregated": aggregated}

    def run(self, task: str, context: Dict[str, Any] = {}) -> Dict[str, Any]:
        return asyncio.run(self.run_async(task, context))

    def _aggregate(self, results: List[Dict]) -> str:
        if self.aggregation == "vote":
            # Retorna el output más frecuente (simplificado)
            outputs = [r.get("output", "") for r in results]
            return max(set(outputs), key=outputs.count)
        return "\n---\n".join([r.get("output", "") for r in results])
```

### Orquestación con LangGraph (Grafo de Estado)

```python
# orchestration/graph_orchestrator.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator


class TeamState(TypedDict):
    task: str
    architect_plan: str
    code: str
    review: dict
    iterations: int
    final_output: str


def create_dev_team_graph(architect, developer, reviewer, max_iterations: int = 3):
    """
    Crea un grafo de estado para el equipo de desarrollo.
    Implementa el patrón Evaluador-Optimizador con reintentos.
    """

    def architect_node(state: TeamState) -> TeamState:
        result = architect.run(state["task"])
        return {**state, "architect_plan": result["output"]}

    def developer_node(state: TeamState) -> TeamState:
        result = developer.run(state["architect_plan"], {"task": state["task"]})
        return {**state, "code": result["output"]}

    def reviewer_node(state: TeamState) -> TeamState:
        review = reviewer.evaluate(state["code"], {"task": state["task"]})
        return {**state, "review": review, "iterations": state.get("iterations", 0) + 1}

    def should_retry(state: TeamState) -> str:
        """Decide si reintentar o finalizar."""
        review = state.get("review", {})
        iterations = state.get("iterations", 0)

        if review.get("approved") or iterations >= 3:
            return "end"
        return "developer"  # Reintenta con el desarrollador

    # Construir el grafo
    workflow = StateGraph(TeamState)
    workflow.add_node("architect", architect_node)
    workflow.add_node("developer", developer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("architect")
    workflow.add_edge("architect", "developer")
    workflow.add_edge("developer", "reviewer")
    workflow.add_conditional_edges("reviewer", should_retry, {"end": END, "developer": "developer"})

    return workflow.compile()
```

---

## 6. Gestión de Memoria y Bases de Datos Vectoriales

### Memoria a corto plazo

```python
# memory/short_term.py
from collections import deque
from typing import List, Dict, Any


class ShortTermMemory:
    """Buffer de contexto para la tarea actual."""

    def __init__(self, max_turns: int = 20):
        self.buffer = deque(maxlen=max_turns)

    def add(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict]:
        return list(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def to_langchain_messages(self):
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        for msg in self.buffer:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages
```

### Memoria a largo plazo con ChromaDB

```python
# memory/long_term.py
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import os


class LongTermMemory:
    """
    Memoria persistente basada en ChromaDB.
    Implementa búsqueda híbrida (vectorial + léxica).
    """

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_dir: str = "./data/chroma",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

    def store(self, content: str, metadata: Dict[str, Any] = {}) -> None:
        """Almacena texto fragmentado con sus embeddings."""
        chunks = self.text_splitter.split_text(content)
        self.vectorstore.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Recupera los k fragmentos más relevantes semánticamente."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def retrieve_with_score(self, query: str, k: int = 5) -> List[Dict]:
        """Recupera fragmentos con puntuación de similitud."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [{"content": doc.page_content, "score": score, "metadata": doc.metadata}
                for doc, score in results]
```

---

## 7. Model Context Protocol (MCP) y Herramientas

### Herramientas de Git

```python
# tools/git_tools.py
from langchain_core.tools import tool
import subprocess
import os


@tool
def create_branch(branch_name: str, base_branch: str = "main") -> str:
    """
    Crea una nueva rama en el repositorio Git.

    Args:
        branch_name: Nombre de la nueva rama (ej: feature/login)
        base_branch: Rama base desde la que crear (default: main)
    Returns:
        Mensaje de éxito o error
    """
    try:
        subprocess.run(["git", "checkout", base_branch], check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-b", branch_name], check=True, capture_output=True)
        return f"✅ Rama '{branch_name}' creada desde '{base_branch}'"
    except subprocess.CalledProcessError as e:
        return f"❌ Error al crear rama: {e.stderr.decode()}"


@tool
def commit_changes(message: str, files: list = None) -> str:
    """
    Realiza un commit con los cambios actuales.

    Args:
        message: Mensaje del commit (convencional: feat/fix/docs/...)
        files: Lista de archivos a incluir (None = todos)
    Returns:
        Hash del commit o mensaje de error
    """
    try:
        if files:
            for f in files:
                subprocess.run(["git", "add", f], check=True, capture_output=True)
        else:
            subprocess.run(["git", "add", "."], check=True, capture_output=True)

        result = subprocess.run(
            ["git", "commit", "-m", message],
            check=True, capture_output=True, text=True
        )
        return f"✅ Commit realizado: {result.stdout.strip()}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error en commit: {e.stderr}"
```

### Herramientas de código

```python
# tools/code_tools.py
from langchain_core.tools import tool
import subprocess
import tempfile
import os


@tool
def run_code(code: str, language: str = "python") -> str:
    """
    Ejecuta un fragmento de código en un entorno aislado.

    Args:
        code: El código a ejecutar
        language: Lenguaje de programación (default: python)
    Returns:
        stdout + stderr de la ejecución
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout or result.stderr
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Timeout: La ejecución superó 30 segundos"
    finally:
        os.unlink(tmp_path)


@tool
def lint_code(code: str) -> str:
    """
    Analiza el código en busca de problemas de estilo y errores.

    Args:
        code: Código Python a analizar
    Returns:
        Reporte de linting (errores y advertencias)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", "-m", "flake8", "--max-line-length=100", tmp_path],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return "✅ Sin errores de linting"
        return f"⚠️ Issues encontrados:\n{result.stdout}"
    finally:
        os.unlink(tmp_path)


@tool
def format_code(code: str) -> str:
    """
    Formatea el código usando Black.

    Args:
        code: Código Python a formatear
    Returns:
        Código formateado
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        subprocess.run(["python", "-m", "black", tmp_path], check=True, capture_output=True)
        with open(tmp_path) as f:
            return f.read()
    except Exception as e:
        return f"❌ Error al formatear: {str(e)}"
    finally:
        os.unlink(tmp_path)
```

---

## 8. Observabilidad y Trazado

### Sistema de trazado asíncrono

```python
# observability/tracer.py
import asyncio
import time
import uuid
import json
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    task: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tool_calls: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    status: str = "running"
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "agent": self.agent_name,
            "task": self.task[:100],
            "latency_ms": round(self.latency_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": len(self.tool_calls),
            "status": self.status,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
        }


class AgentTracer:
    """Trazado distribuido asíncrono para agentes."""

    _traces: list = []
    _send_queue: asyncio.Queue = None

    @classmethod
    def trace(cls, agent_name: str):
        """Decorador para trazar la ejecución de un agente."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                trace = Trace(agent_name=agent_name)
                try:
                    result = func(*args, **kwargs)
                    trace.status = "completed"
                    return result
                except Exception as e:
                    trace.status = "failed"
                    trace.error = str(e)
                    raise
                finally:
                    trace.end_time = time.time()
                    cls._traces.append(trace)
                    logger.info(f"TRACE: {json.dumps(trace.to_dict())}")
            return wrapper
        return decorator

    @classmethod
    def get_summary(cls) -> Dict:
        """Retorna un resumen de todas las trazas registradas."""
        if not cls._traces:
            return {}
        latencies = [t.latency_ms for t in cls._traces if t.end_time]
        return {
            "total_traces": len(cls._traces),
            "completed": sum(1 for t in cls._traces if t.status == "completed"),
            "failed": sum(1 for t in cls._traces if t.status == "failed"),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 2),
            "total_tokens": sum(t.tokens_in + t.tokens_out for t in cls._traces),
        }
```

### Métricas de KPIs

```python
# observability/metrics.py
from dataclasses import dataclass, field
from typing import List
import time


@dataclass
class AgentMetrics:
    """Registra KPIs operativos del agente."""

    total_tasks: int = 0
    completed_without_human: int = 0
    escalated_to_human: int = 0
    resolved_first_attempt: int = 0
    resolution_times: List[float] = field(default_factory=list)
    accuracy_scores: List[float] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        """TCR: % tareas completadas sin intervención humana. Objetivo: >85%"""
        if self.total_tasks == 0:
            return 0.0
        return round(self.completed_without_human / self.total_tasks * 100, 2)

    @property
    def containment_rate(self) -> float:
        """% consultas resueltas sin escalamiento."""
        total = self.completed_without_human + self.escalated_to_human
        if total == 0:
            return 0.0
        return round(self.completed_without_human / total * 100, 2)

    @property
    def first_contact_resolution(self) -> float:
        """% problemas resueltos en primer intento. Objetivo: >70%"""
        if self.total_tasks == 0:
            return 0.0
        return round(self.resolved_first_attempt / self.total_tasks * 100, 2)

    @property
    def avg_resolution_time_s(self) -> float:
        if not self.resolution_times:
            return 0.0
        return round(sum(self.resolution_times) / len(self.resolution_times), 2)

    @property
    def avg_accuracy(self) -> float:
        if not self.accuracy_scores:
            return 0.0
        return round(sum(self.accuracy_scores) / len(self.accuracy_scores), 2)

    def record_task(
        self, completed: bool, escalated: bool, first_attempt: bool,
        duration_s: float, accuracy: float
    ) -> None:
        self.total_tasks += 1
        if completed and not escalated:
            self.completed_without_human += 1
        if escalated:
            self.escalated_to_human += 1
        if first_attempt:
            self.resolved_first_attempt += 1
        self.resolution_times.append(duration_s)
        self.accuracy_scores.append(accuracy)

    def report(self) -> dict:
        return {
            "task_completion_rate_%": self.task_completion_rate,
            "containment_rate_%": self.containment_rate,
            "first_contact_resolution_%": self.first_contact_resolution,
            "avg_resolution_time_s": self.avg_resolution_time_s,
            "avg_accuracy_%": self.avg_accuracy,
            "total_tasks": self.total_tasks,
        }
```

---

## 9. Ciclo de Vida del Desarrollo Agéntico (ADLC)

### Registro centralizado de prompts

```python
# governance/prompt_registry.py
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class PromptRegistry:
    """
    Registro centralizado de prompts con control de versiones.
    Formato de clave: {feature}-{purpose}-{version}
    Ejemplo: auth-system-validator-v2
    """

    def __init__(self, registry_path: str = "./config/prompts.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register(self, key: str, prompt: str, author: str, description: str = "") -> str:
        """Registra o actualiza un prompt con versionado automático."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]

        if key not in self.registry:
            self.registry[key] = {"versions": [], "current": None}

        version_entry = {
            "hash": prompt_hash,
            "prompt": prompt,
            "author": author,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "version_number": len(self.registry[key]["versions"]) + 1,
        }

        self.registry[key]["versions"].append(version_entry)
        self.registry[key]["current"] = prompt_hash
        self._save()
        return prompt_hash

    def get(self, key: str, version_hash: Optional[str] = None) -> Optional[str]:
        """Obtiene un prompt por su clave y versión (default: actual)."""
        if key not in self.registry:
            return None
        target_hash = version_hash or self.registry[key]["current"]
        for v in self.registry[key]["versions"]:
            if v["hash"] == target_hash:
                return v["prompt"]
        return None

    def list_versions(self, key: str) -> list:
        """Lista todas las versiones de un prompt."""
        if key not in self.registry:
            return []
        return [
            {"hash": v["hash"], "version": v["version_number"],
             "author": v["author"], "created_at": v["created_at"]}
            for v in self.registry[key]["versions"]
        ]
```

---

## 10. Estrategias de Manejo de Errores

### Bucle de autocorrección con reflexión

```python
# agents/self_correcting_agent.py
from agents.base_agent import BaseAgent
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SelfCorrectingMixin:
    """
    Mixin que agrega capacidad de autocorrección a cualquier agente.
    Implementa el patrón: Reflexión → Planificación → Reintento
    """

    max_retries: int = 3

    def run_with_correction(
        self,
        task: str,
        context: Dict[str, Any] = {},
        validator: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta la tarea con reintentos automáticos en caso de fallo.

        Args:
            task: La tarea a ejecutar
            context: Contexto adicional
            validator: Función que retorna (bool, str) → (válido, mensaje_error)
        """
        errors = []
        last_result = None

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"[{self.name}] Intento {attempt}/{self.max_retries}")

            try:
                result = self.run(task, context)
                last_result = result

                # Validar resultado si hay validador
                if validator:
                    is_valid, error_msg = validator(result["output"])
                    if not is_valid:
                        errors.append(f"Intento {attempt}: {error_msg}")
                        # Reflexión: incluir el error como contexto para el reintento
                        context["previous_errors"] = errors
                        context["reflection"] = (
                            f"Tu respuesta anterior falló por: {error_msg}. "
                            f"Corrige este problema específico."
                        )
                        continue

                result["attempts"] = attempt
                result["errors"] = errors
                return result

            except Exception as e:
                error_msg = str(e)
                errors.append(f"Intento {attempt} - Excepción: {error_msg}")
                context["previous_errors"] = errors
                logger.warning(f"[{self.name}] Error en intento {attempt}: {error_msg}")

        # Agotados los reintentos: escalar a humano
        logger.error(f"[{self.name}] Máximo de reintentos alcanzado. Escalando a humano.")
        return {
            "output": last_result.get("output", "") if last_result else "",
            "status": "escalated",
            "attempts": self.max_retries,
            "errors": errors,
            "requires_human_review": True,
        }
```

---

## 11. Testing y Evaluación

### Tests unitarios

```python
# tests/unit/test_agents.py
import pytest
from unittest.mock import MagicMock, patch
from agents.architect_agent import ArchitectAgent
from agents.reviewer_agent import ReviewerAgent


class TestArchitectAgent:

    @pytest.fixture
    def architect(self):
        with patch("langchain_openai.ChatOpenAI"):
            agent = ArchitectAgent(model="gpt-4o")
            agent.chain = MagicMock()
            return agent

    def test_agent_initialization(self, architect):
        assert architect.name == "Arquitecto"
        assert architect.role is not None
        assert architect.agent_id is not None

    def test_perceive_updates_context(self, architect):
        input_data = {"requirements": "Build login system", "tech_stack": "FastAPI"}
        context = architect.perceive(input_data)
        assert context["requirements"] == "Build login system"
        assert context["tech_stack"] == "FastAPI"

    def test_reason_calls_chain(self, architect):
        architect.chain.invoke = MagicMock(return_value="Technical plan here")
        result = architect.reason("Build auth", {"requirements": "login"})
        architect.chain.invoke.assert_called_once()
        assert result == "Technical plan here"

    def test_full_run_cycle(self, architect):
        architect.chain.invoke = MagicMock(return_value="Plan: 1. Setup, 2. Implement")
        result = architect.run("Create API", {"stack": "FastAPI"})
        assert result["status"] == "completed"
        assert "output" in result
        assert architect.state.status == "completed"


class TestReviewerAgent:

    @pytest.fixture
    def reviewer(self):
        with patch("langchain_openai.ChatOpenAI"):
            agent = ReviewerAgent(model="gpt-4o")
            agent.chain = MagicMock()
            return agent

    def test_evaluate_returns_structured_output(self, reviewer):
        mock_response = '{"approved": true, "score": 85, "issues": [], "requires_retry": false}'
        reviewer.chain.invoke = MagicMock(return_value=mock_response)

        result = reviewer.evaluate("def foo(): pass", {"task": "simple function"})
        assert isinstance(result, dict)
        assert "approved" in result
        assert result["approved"] is True

    def test_evaluate_handles_invalid_json(self, reviewer):
        reviewer.chain.invoke = MagicMock(return_value="invalid json response")
        result = reviewer.evaluate("code", {})
        assert result["approved"] is False
        assert result["requires_retry"] is True
```

### Tests de integración del orquestador

```python
# tests/integration/test_orchestration.py
import pytest
from unittest.mock import MagicMock, patch
from orchestration.sequential import SequentialOrchestrator


class TestSequentialOrchestrator:

    def _make_mock_agent(self, name: str, output: str):
        agent = MagicMock()
        agent.name = name
        agent.run.return_value = {"output": output, "status": "completed"}
        return agent

    def test_sequential_passes_output_as_next_input(self):
        agent_a = self._make_mock_agent("Architect", "Plan: build API")
        agent_b = self._make_mock_agent("Developer", "Code: def api(): ...")

        orchestrator = SequentialOrchestrator(agents=[agent_a, agent_b])
        result = orchestrator.run("Build a REST API")

        # El segundo agente debe recibir el output del primero como tarea
        agent_b.run.assert_called_once()
        call_args = agent_b.run.call_args
        assert "Plan: build API" in call_args[0][0]

    def test_sequential_returns_final_output(self):
        agent_a = self._make_mock_agent("A", "step 1")
        agent_b = self._make_mock_agent("B", "step 2")
        agent_c = self._make_mock_agent("C", "final result")

        orchestrator = SequentialOrchestrator(agents=[agent_a, agent_b, agent_c])
        result = orchestrator.run("task")

        assert result["final_output"] == "final result"
        assert len(result["results"]) == 3
```

### Evaluador LLM-as-a-Judge

```python
# evaluation/llm_judge.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict


JUDGE_PROMPT = """
Eres un evaluador objetivo de sistemas de IA. Evalúa la siguiente respuesta del agente.
IMPORTANTE: No conoces qué agente generó esta respuesta. Evalúa solo la calidad.

Tarea original: {task}
Respuesta del agente: {response}
Criterios de evaluación: {criteria}

Devuelve SOLO un JSON con esta estructura:
{{
  "relevance_score": 0-10,
  "accuracy_score": 0-10,
  "safety_score": 0-10,
  "hallucination_detected": true/false,
  "overall_score": 0-10,
  "reasoning": "explicación breve"
}}
"""


class LLMJudge:
    """Evaluador automático usando un LLM como juez."""

    def __init__(self, judge_model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=judge_model, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        self.chain = self.prompt | self.llm | JsonOutputParser()

    def evaluate(self, task: str, response: str, criteria: str = "accuracy, relevance, safety") -> Dict:
        try:
            return self.chain.invoke({
                "task": task,
                "response": response,
                "criteria": criteria,
            })
        except Exception as e:
            return {"overall_score": 0, "reasoning": f"Evaluation failed: {str(e)}"}


# tests/evaluation/test_golden_dataset.py
import pytest
from evaluation.llm_judge import LLMJudge


GOLDEN_DATASET = [
    {
        "task": "Create a Python function to calculate fibonacci",
        "expected_keywords": ["def", "fibonacci", "return"],
        "min_score": 7,
    },
    {
        "task": "Explain what a REST API is",
        "expected_keywords": ["HTTP", "endpoint", "request"],
        "min_score": 7,
    },
]


class TestGoldenDataset:

    @pytest.fixture
    def judge(self):
        return LLMJudge()

    @pytest.mark.parametrize("example", GOLDEN_DATASET)
    def test_response_meets_quality_threshold(self, judge, example):
        """Test de regresión: las respuestas deben superar el umbral mínimo de calidad."""
        # En tests reales, aquí irían las respuestas del agente en producción
        mock_response = f"Here is information about: {example['task']}"
        result = judge.evaluate(example["task"], mock_response)
        assert isinstance(result.get("overall_score"), (int, float))
```

### Ejecutar todos los tests

```bash
# Ejecutar suite completa
pytest tests/ -v --tb=short

# Solo tests unitarios (rápido, sin LLM)
pytest tests/unit/ -v

# Tests con cobertura
pytest tests/ --cov=agents --cov=orchestration --cov-report=html

# Tests de evaluación (requieren API key real)
pytest tests/evaluation/ -v -m "not expensive"
```

---

## 12. KPIs y Medición de ROI

### Dashboard de métricas

```python
# main.py - Ejemplo de uso completo con métricas
from observability.metrics import AgentMetrics
from observability.tracer import AgentTracer
import time


def run_team_with_metrics():
    metrics = AgentMetrics()

    # Simular ejecución de tareas (reemplazar con agentes reales)
    tasks = [
        {"task": "Implement login API", "expected_time": 30},
        {"task": "Write unit tests for auth module", "expected_time": 20},
        {"task": "Review PR #142", "expected_time": 10},
    ]

    for task_def in tasks:
        start = time.time()

        # Aquí iría: result = orchestrator.run(task_def["task"])
        # Simulamos para el ejemplo:
        success = True
        escalated = False
        first_attempt = True

        duration = time.time() - start
        metrics.record_task(
            completed=success,
            escalated=escalated,
            first_attempt=first_attempt,
            duration_s=duration,
            accuracy=0.92,
        )

    # Imprimir reporte
    report = metrics.report()
    print("\n📊 REPORTE DE KPIs DEL EQUIPO AGÉNTICO")
    print("=" * 45)
    for key, value in report.items():
        status = ""
        if "completion_rate" in key and value > 85:
            status = "✅"
        elif "first_contact" in key and value > 70:
            status = "✅"
        else:
            status = "📈"
        print(f"{status} {key}: {value}")

    # Calcular ROI simplificado
    hourly_rate = 50  # USD/hora desarrollador
    hours_saved_per_task = 2
    tasks_per_month = 100
    agent_cost_monthly = 500  # USD (API + infra)

    monthly_savings = tasks_per_month * hours_saved_per_task * hourly_rate
    roi = ((monthly_savings - agent_cost_monthly) / agent_cost_monthly) * 100

    print(f"\n💰 ROI ESTIMADO MENSUAL")
    print(f"  Ahorro bruto:   ${monthly_savings:,}")
    print(f"  Costo agente:   ${agent_cost_monthly:,}")
    print(f"  ROI:            {roi:.1f}%")


if __name__ == "__main__":
    run_team_with_metrics()
```

---

## Resumen de la Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                   │
│        Sequential | Concurrent | LangGraph               │
└──────────────┬──────────────────────────────┬───────────┘
               │                              │
   ┌───────────▼──────────┐    ┌─────────────▼────────────┐
   │   ARCHITECT AGENT    │    │    DEVELOPER AGENT        │
   │  (Plan & Design)     │───▶│  (Code & Execute)         │
   └──────────────────────┘    └─────────────┬────────────┘
                                             │
                               ┌─────────────▼────────────┐
                               │    REVIEWER AGENT         │
                               │  (Evaluate & Retry)       │
                               └─────────────┬────────────┘
                                             │
         ┌───────────────────────────────────▼──────────┐
         │              SHARED INFRASTRUCTURE            │
         │  Memory (ST + LT/Chroma) | Tools | Tracer     │
         │  Prompt Registry | Metrics | LLM Judge        │
         └───────────────────────────────────────────────┘
```

> **Siguiente paso recomendado:** Comenzar con el `ArchitectAgent` y el `ReviewerAgent` en un pipeline secuencial simple, añadir observabilidad desde el primer día, y escalar progresivamente hacia la orquestación con LangGraph.
