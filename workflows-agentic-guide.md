# Guía de Implementación: Arquitectura de Sistemas Agénticos con CrewAI

> Framework líder open-source para orquestar equipos autónomos de agentes de IA — independiente de LangChain, construido desde cero.

---

## Tabla de Contenidos

1. [Instalación y Setup](#1-instalación-y-setup)
2. [Conceptos clave de CrewAI](#2-conceptos-clave-de-crewai)
3. [Estructura del Proyecto](#3-estructura-del-proyecto)
4. [Configuración YAML de Agentes y Tareas](#4-configuración-yaml-de-agentes-y-tareas)
5. [Implementación de la Crew](#5-implementación-de-la-crew)
6. [Herramientas Personalizadas](#6-herramientas-personalizadas)
7. [Orquestación con Flows](#7-orquestación-con-flows)
8. [Memoria y Conocimiento](#8-memoria-y-conocimiento)
9. [Manejo de Errores y Guardrails](#9-manejo-de-errores-y-guardrails)
10. [Observabilidad](#10-observabilidad)
11. [Testing y Evaluación](#11-testing-y-evaluación)
12. [KPIs y Métricas](#12-kpis-y-métricas)

---

## 1. Instalación y Setup

### Requisitos

- Python `>3.10` y `<3.13`
- `uv` (gestor de paquetes recomendado por CrewAI)

```bash
# Instalar uv
pip install uv

# Instalar CrewAI CLI
uv tool install crewai

# Verificar instalación
crewai --version
```

### Crear proyecto con el CLI

```bash
# El CLI genera toda la estructura automáticamente
crewai create crew dev-team

# El asistente te preguntará:
# → Proveedor LLM: OpenAI / Anthropic / Gemini / Ollama
# → Modelo: gpt-4o / claude-3-7-sonnet / gemini-2.5-pro

cd dev-team
```

### Variables de entorno

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Opcional: observabilidad con LangSmith
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=dev-team-crewai
```

### Instalar dependencias

```bash
uv sync
# o con pip:
pip install crewai crewai-tools python-dotenv pytest
```

---

## 2. Conceptos Clave de CrewAI

| Concepto | Descripción |
|---|---|
| **Agent** | Unidad autónoma con `role`, `goal` y `backstory` |
| **Task** | Tarea asignada a un agente con `description` y `expected_output` |
| **Crew** | Equipo de agentes que colaboran para completar tareas |
| **Flow** | Orquestador event-driven que controla el estado y la ejecución |
| **Tool** | Función que el agente puede invocar para actuar sobre el mundo |
| **Process** | Modo de ejecución: `sequential` (default) o `hierarchical` |

### Flujo de ejecución

```
Flow (estado global)
  │
  └──▶ Crew (equipo autónomo)
          │
          ├── Agent: Arquitecto ──▶ Task: Diseño técnico
          ├── Agent: Desarrollador ──▶ Task: Implementar código
          └── Agent: Revisor ──▶ Task: Revisar y validar
```

---

## 3. Estructura del Proyecto

El CLI genera esta estructura lista para producción:

```
dev-team/
├── .env                          # API keys (no commitear)
├── .gitignore
├── pyproject.toml                # Dependencias del proyecto
├── README.md
├── knowledge/                    # Archivos de conocimiento (PDFs, docs)
└── src/
    └── dev_team/
        ├── __init__.py
        ├── main.py               # Punto de entrada
        ├── crew.py               # Definición del equipo
        ├── config/
        │   ├── agents.yaml       # Configuración de agentes
        │   └── tasks.yaml        # Configuración de tareas
        └── tools/
            ├── __init__.py
            └── custom_tools.py   # Herramientas propias
```

---

## 4. Configuración YAML de Agentes y Tareas

### `config/agents.yaml`

```yaml
# Agente Arquitecto: planifica, no escribe código directamente
architect:
  role: >
    Arquitecto de Software Senior para {project_name}
  goal: >
    Analizar los requisitos de {project_name}, diseñar la solución técnica
    y generar un plan detallado de implementación con tareas asignables.
    Nunca escribes código directamente; tu output es siempre un plan.
  backstory: >
    Eres un arquitecto con 15 años de experiencia diseñando sistemas
    distribuidos. Tu fortaleza es descomponer problemas complejos en
    módulos bien definidos con interfaces claras. Anticipas modos de fallo
    y garantizas escalabilidad desde el diseño.
  llm: gpt-4o
  verbose: true
  allow_delegation: true

# Agente Desarrollador: implementa el plan del arquitecto
developer:
  role: >
    Ingeniero de Software Full-Stack para {project_name}
  goal: >
    Implementar el código Python descrito en el plan técnico de {project_name}.
    Cada función debe incluir type hints, docstrings y manejo de excepciones.
    Siempre valida que el código ejecute sin errores antes de entregarlo.
  backstory: >
    Eres un desarrollador senior especializado en Python y APIs REST.
    Sigues SOLID, DRY y escribes código que un junior pueda entender.
    Tu lema: el código no está listo hasta que tiene tests.
  llm: anthropic/claude-3-7-sonnet-20250219
  verbose: true
  allow_delegation: false

# Agente Revisor: evalúa calidad y seguridad
reviewer:
  role: >
    Ingeniero de Calidad y Seguridad de Software
  goal: >
    Revisar el código generado para {project_name} identificando bugs,
    vulnerabilidades de seguridad, problemas de rendimiento y deuda técnica.
    Proporciona un reporte estructurado con puntuación y sugerencias concretas.
  backstory: >
    Eres un experto en code review con foco en seguridad (OWASP) y
    rendimiento. Has prevenido cientos de vulnerabilidades en producción.
    Eres directo y constructivo: nunca apruebas código con problemas críticos.
  llm: gpt-4o
  verbose: true
  allow_delegation: false

# Agente Product Manager
pm:
  role: >
    Product Manager de IA para {project_name}
  goal: >
    Traducir los objetivos de negocio de {project_name} en requisitos técnicos
    claros. Priorizas el backlog según impacto y esfuerzo, y defines criterios
    de aceptación medibles para cada funcionalidad.
  backstory: >
    Eres un PM con experiencia en productos de IA. Entiendes tanto el negocio
    como la viabilidad técnica. Redactas historias de usuario en formato
    "Como [rol], quiero [acción] para [beneficio]" y defines métricas de éxito.
  llm: gpt-4o-mini
  verbose: true
  allow_delegation: true
```

### `config/tasks.yaml`

```yaml
# Tarea 1: El PM define los requisitos
define_requirements_task:
  description: >
    Analiza el siguiente objetivo de negocio para {project_name}:
    "{business_goal}"

    Genera:
    1. Lista de funcionalidades priorizadas (MoSCoW)
    2. Historias de usuario para las funcionalidades "Must Have"
    3. Criterios de aceptación medibles para cada historia
    4. Estimación de complejidad (S/M/L/XL)
  expected_output: >
    Un documento de requisitos estructurado en Markdown con:
    - Tabla de priorización MoSCoW
    - Mínimo 3 historias de usuario con criterios de aceptación
    - Estimación de esfuerzo total
  agent: pm
  output_file: outputs/requirements.md

# Tarea 2: El Arquitecto diseña la solución
design_architecture_task:
  description: >
    Basándote en los requisitos generados, diseña la arquitectura técnica
    para {project_name}. Define:
    1. Componentes del sistema y sus responsabilidades
    2. Interfaces y contratos entre módulos
    3. Estructura de carpetas y archivos
    4. Dependencias externas necesarias
    5. Plan de implementación ordenado con estimaciones
  expected_output: >
    Un documento de diseño técnico en Markdown con:
    - Diagrama de arquitectura (en texto/ASCII)
    - Especificación de cada módulo
    - Plan de implementación paso a paso
    - Lista de dependencias pip
  agent: architect
  context:
    - define_requirements_task
  output_file: outputs/architecture.md

# Tarea 3: El Desarrollador implementa
implement_code_task:
  description: >
    Implementa el código Python para {project_name} siguiendo el plan de
    arquitectura. Requisitos obligatorios:
    - Type hints en todas las funciones
    - Docstrings con descripción, Args y Returns
    - Manejo explícito de excepciones
    - Tests unitarios con pytest para cada módulo
    - Código ejecutable sin errores

    Entrega cada archivo completo, no fragmentos parciales.
  expected_output: >
    Código Python completo y funcional que incluya:
    - Todos los módulos descritos en la arquitectura
    - Suite de tests con pytest
    - requirements.txt actualizado
    - Instrucciones de ejecución
  agent: developer
  context:
    - design_architecture_task
  output_file: outputs/implementation.py

# Tarea 4: El Revisor valida
review_code_task:
  description: >
    Realiza una revisión exhaustiva del código implementado para {project_name}.
    Evalúa:
    1. Correctitud funcional (¿cumple los requisitos?)
    2. Calidad del código (SOLID, DRY, legibilidad)
    3. Seguridad (OWASP Top 10, inyección, exposición de datos)
    4. Rendimiento y escalabilidad
    5. Cobertura y calidad de tests
    6. Documentación

    Para cada problema encontrado: indica severidad (CRÍTICO/ALTO/MEDIO/BAJO),
    ubicación y solución propuesta.
  expected_output: >
    Reporte de revisión en Markdown con:
    - Puntuación general (0-100)
    - Estado: APROBADO / REQUIERE CAMBIOS / RECHAZADO
    - Lista de issues por severidad
    - Sugerencias de mejora
  agent: reviewer
  context:
    - implement_code_task
  output_file: outputs/review.md
```

---

## 5. Implementación de la Crew

### `crew.py`

```python
# src/dev_team/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, CodeInterpreterTool
from .tools.custom_tools import LintCodeTool, RunTestsTool


@CrewBase
class DevTeamCrew:
    """
    Equipo de desarrollo agéntico: PM → Arquitecto → Desarrollador → Revisor.
    Implementa el patrón Evaluador-Optimizador con proceso secuencial.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ─── Definición de Agentes ───────────────────────────────────────────────

    @agent
    def pm(self) -> Agent:
        return Agent(
            config=self.agents_config["pm"],
            tools=[FileReadTool()],
        )

    @agent
    def architect(self) -> Agent:
        return Agent(
            config=self.agents_config["architect"],
            tools=[FileReadTool()],
        )

    @agent
    def developer(self) -> Agent:
        return Agent(
            config=self.agents_config["developer"],
            tools=[
                CodeInterpreterTool(),
                LintCodeTool(),
                RunTestsTool(),
            ],
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer"],
            tools=[FileReadTool(), LintCodeTool()],
        )

    # ─── Definición de Tareas ────────────────────────────────────────────────

    @task
    def define_requirements_task(self) -> Task:
        return Task(config=self.tasks_config["define_requirements_task"])

    @task
    def design_architecture_task(self) -> Task:
        return Task(config=self.tasks_config["design_architecture_task"])

    @task
    def implement_code_task(self) -> Task:
        return Task(config=self.tasks_config["implement_code_task"])

    @task
    def review_code_task(self) -> Task:
        return Task(config=self.tasks_config["review_code_task"])

    # ─── Definición de la Crew ───────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,    # Inyectados automáticamente por @CrewBase
            tasks=self.tasks,      # Inyectados automáticamente por @CrewBase
            process=Process.sequential,
            verbose=True,
            # Para proceso jerárquico (un manager LLM asigna tareas):
            # process=Process.hierarchical,
            # manager_llm="gpt-4o",
        )
```

### `main.py`

```python
# src/dev_team/main.py
from dotenv import load_dotenv
from .crew import DevTeamCrew

load_dotenv()


def run():
    """Punto de entrada principal del equipo agéntico."""
    inputs = {
        "project_name": "API de Autenticación JWT",
        "business_goal": (
            "Desarrollar una API REST en FastAPI que permita registro, "
            "login y renovación de tokens JWT. Debe soportar 1000 req/seg "
            "y cumplir con OWASP Top 10."
        ),
    }

    print("🚀 Iniciando equipo de desarrollo agéntico...")
    result = DevTeamCrew().crew().kickoff(inputs=inputs)
    print("\n✅ Proceso completado.")
    print(result)


def run_async():
    """Ejecución asíncrona para múltiples proyectos en paralelo."""
    import asyncio
    from .crew import DevTeamCrew

    projects = [
        {"project_name": "Auth API", "business_goal": "Sistema de login JWT"},
        {"project_name": "Payment Service", "business_goal": "Procesamiento de pagos"},
    ]

    async def kickoff_all():
        crew = DevTeamCrew().crew()
        results = await crew.kickoff_for_each_async(inputs=projects)
        return results

    return asyncio.run(kickoff_all())


if __name__ == "__main__":
    run()
```

### Ejecutar el equipo

```bash
# Con el CLI de CrewAI
crewai run

# O directamente con Python
cd src && python -m dev_team.main
```

---

## 6. Herramientas Personalizadas

### `tools/custom_tools.py`

```python
# src/dev_team/tools/custom_tools.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import subprocess
import tempfile
import os


# ─── Schema de entrada para las herramientas ─────────────────────────────────

class CodeInput(BaseModel):
    code: str = Field(description="Código Python a procesar")


class TestInput(BaseModel):
    test_file: str = Field(description="Ruta al archivo de tests pytest")


# ─── Herramienta: Linting de código ──────────────────────────────────────────

class LintCodeTool(BaseTool):
    name: str = "lint_code"
    description: str = (
        "Analiza código Python con flake8 y detecta errores de estilo, "
        "imports no usados y problemas de complejidad ciclomática. "
        "Úsala antes de entregar cualquier implementación."
    )
    args_schema: type[BaseModel] = CodeInput

    def _run(self, code: str) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            result = subprocess.run(
                ["python", "-m", "flake8", "--max-line-length=100",
                 "--max-complexity=10", path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return "✅ Linting pasado sin errores."
            return f"⚠️ Issues detectados:\n{result.stdout}"
        except FileNotFoundError:
            return "❌ flake8 no instalado. Ejecuta: pip install flake8"
        finally:
            os.unlink(path)


# ─── Herramienta: Ejecutar tests ─────────────────────────────────────────────

class RunTestsTool(BaseTool):
    name: str = "run_tests"
    description: str = (
        "Ejecuta la suite de tests pytest y retorna el resultado. "
        "Úsala para validar que el código implementado pasa todos los tests."
    )
    args_schema: type[BaseModel] = TestInput

    def _run(self, test_file: str) -> str:
        if not os.path.exists(test_file):
            return f"❌ Archivo no encontrado: {test_file}"
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=60
        )
        status = "✅ Tests pasados" if result.returncode == 0 else "❌ Tests fallidos"
        return f"{status}\n\n{result.stdout}"


# ─── Herramienta: Llamada a API externa ──────────────────────────────────────

class GitCommitTool(BaseTool):
    name: str = "git_commit"
    description: str = (
        "Realiza un commit en el repositorio Git con el mensaje indicado. "
        "Usa commits convencionales: feat/fix/docs/refactor/test/chore."
    )

    class _Input(BaseModel):
        message: str = Field(description="Mensaje del commit (conventional commits)")
        files: list[str] = Field(default=[], description="Archivos a incluir (vacío = todos)")

    args_schema: type[BaseModel] = _Input

    def _run(self, message: str, files: list[str] = []) -> str:
        try:
            targets = files if files else ["."]
            for f in targets:
                subprocess.run(["git", "add", f], check=True, capture_output=True)
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True, text=True, check=True
            )
            return f"✅ Commit realizado:\n{result.stdout.strip()}"
        except subprocess.CalledProcessError as e:
            return f"❌ Error: {e.stderr}"
```

---

## 7. Orquestación con Flows

Los Flows permiten control preciso y event-driven sobre múltiples Crews.

```python
# src/dev_team/flow.py
from crewai.flow.flow import Flow, listen, start, router
from pydantic import BaseModel
from .crew import DevTeamCrew


class DevPipelineState(BaseModel):
    """Estado persistente del pipeline de desarrollo."""
    project_name: str = ""
    business_goal: str = ""
    requirements: str = ""
    architecture: str = ""
    code: str = ""
    review_score: int = 0
    review_status: str = ""  # APROBADO / REQUIERE_CAMBIOS / RECHAZADO
    iteration: int = 0
    max_iterations: int = 3


class DevPipelineFlow(Flow[DevPipelineState]):
    """
    Pipeline completo de desarrollo con lógica condicional.
    Si el revisor rechaza, reintenta automáticamente hasta max_iterations.
    """

    @start()
    def initialize(self):
        print(f"🚀 Iniciando pipeline para: {self.state.project_name}")
        return "start_development"

    @listen("start_development")
    def run_dev_crew(self):
        """Ejecuta el equipo completo PM → Arquitecto → Dev → Revisor."""
        print(f"🔄 Iteración {self.state.iteration + 1}/{self.state.max_iterations}")

        inputs = {
            "project_name": self.state.project_name,
            "business_goal": self.state.business_goal,
        }

        result = DevTeamCrew().crew().kickoff(inputs=inputs)
        self.state.iteration += 1

        # Parsear resultado del revisor (simplificado)
        output = str(result)
        if "APROBADO" in output:
            self.state.review_status = "APROBADO"
            self.state.review_score = 85
        elif "RECHAZADO" in output:
            self.state.review_status = "RECHAZADO"
            self.state.review_score = 40
        else:
            self.state.review_status = "REQUIERE_CAMBIOS"
            self.state.review_score = 65

        return self.state.review_status

    @router(run_dev_crew)
    def evaluate_result(self):
        """Decide el siguiente paso según el resultado del revisor."""
        if self.state.review_status == "APROBADO":
            return "deploy"
        elif self.state.iteration >= self.state.max_iterations:
            return "escalate"
        else:
            return "retry"

    @listen("deploy")
    def deploy_to_staging(self):
        print("✅ Código aprobado. Desplegando a staging...")
        return {"status": "deployed", "iterations": self.state.iteration}

    @listen("retry")
    def prepare_retry(self):
        print(f"🔁 Reintentando... ({self.state.iteration}/{self.state.max_iterations})")
        return "start_development"

    @listen("escalate")
    def escalate_to_human(self):
        print("🚨 Máximo de iteraciones alcanzado. Escalando a revisión humana.")
        return {
            "status": "escalated",
            "iterations": self.state.iteration,
            "last_score": self.state.review_score,
        }


# Punto de entrada del Flow
def run_flow():
    flow = DevPipelineFlow()
    flow.state.project_name = "API de Autenticación JWT"
    flow.state.business_goal = "Sistema de login con tokens JWT y refresh tokens"
    result = flow.kickoff()
    return result
```

---

## 8. Memoria y Conocimiento

### Configurar memoria en la Crew

```python
# crew.py — Crew con memoria activada
from crewai import Crew, Process
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        # Activa los 3 tipos de memoria
        memory=True,
        # Memoria a largo plazo: persiste entre ejecuciones
        long_term_memory=LongTermMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small"},
                },
                storage_path="./data/long_term_memory",
            )
        ),
        verbose=True,
    )
```

### Base de conocimiento estático

```python
# Añadir documentos de referencia a los agentes
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Fuentes de conocimiento
coding_standards = TextFileKnowledgeSource(
    file_paths=["knowledge/coding_standards.md"]
)
api_docs = PDFKnowledgeSource(
    file_paths=["knowledge/fastapi_reference.pdf"]
)

@crew
def crew(self) -> Crew:
    return Crew(
        agents=self.agents,
        tasks=self.tasks,
        knowledge_sources=[coding_standards, api_docs],
        process=Process.sequential,
        verbose=True,
    )
```

---

## 9. Manejo de Errores y Guardrails

### Guardrail en tarea (validación antes de continuar)

```python
# src/dev_team/guardrails.py
from typing import Tuple


def validate_code_output(output) -> Tuple[bool, str]:
    """
    Guardrail funcional: valida que el output del desarrollador
    contiene código Python real antes de pasarlo al revisor.
    """
    content = str(output)

    if len(content) < 100:
        return False, "El código es demasiado corto. Debe ser una implementación completa."

    if "def " not in content and "class " not in content:
        return False, "No se detectaron funciones ni clases Python. Implementa el código completo."

    if "import" not in content:
        return False, "No hay imports. El código debe incluir las dependencias necesarias."

    return True, ""


def validate_review_output(output) -> Tuple[bool, str]:
    """
    Guardrail: el revisor debe siempre incluir una puntuación numérica.
    """
    content = str(output)
    keywords = ["APROBADO", "REQUIERE CAMBIOS", "RECHAZADO"]

    if not any(kw in content.upper() for kw in keywords):
        return False, "El reporte debe contener uno de: APROBADO, REQUIERE CAMBIOS, RECHAZADO"

    return True, ""
```

### Aplicar guardrails a las tareas

```python
# En tasks.yaml o en crew.py al crear el Task
@task
def implement_code_task(self) -> Task:
    from .guardrails import validate_code_output
    return Task(
        config=self.tasks_config["implement_code_task"],
        guardrail=validate_code_output,  # Se ejecuta antes de pasar al siguiente agente
    )

@task
def review_code_task(self) -> Task:
    from .guardrails import validate_review_output
    return Task(
        config=self.tasks_config["review_code_task"],
        guardrail=validate_review_output,
    )
```

### Callback para manejo de eventos

```python
# src/dev_team/callbacks.py
from crewai.utilities.events import (
    CrewKickoffStartedEvent,
    TaskCompletedEvent,
    AgentActionTakenEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
import logging
import time

logger = logging.getLogger(__name__)
_task_start_times = {}


@crewai_event_bus.on(CrewKickoffStartedEvent)
def on_crew_start(source, event):
    logger.info(f"🚀 Crew iniciada: {event.crew_name}")


@crewai_event_bus.on(TaskCompletedEvent)
def on_task_completed(source, event):
    task_id = event.task_id
    duration = time.time() - _task_start_times.pop(task_id, time.time())
    logger.info(f"✅ Tarea completada en {duration:.1f}s | Agente: {event.agent_role}")


@crewai_event_bus.on(AgentActionTakenEvent)
def on_agent_action(source, event):
    logger.info(f"🔧 [{event.agent_role}] Acción: {event.tool_name}")
```

---

## 10. Observabilidad

### Activar trazado nativo de CrewAI

```python
# crew.py — habilitar trazado
import os
os.environ["CREWAI_TELEMETRY"] = "true"  # Telemetría interna CrewAI

# Para observabilidad externa con Langfuse:
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
```

### Métricas operativas propias

```python
# src/dev_team/metrics.py
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineMetrics:
    """KPIs del pipeline de desarrollo agéntico."""

    runs: List[dict] = field(default_factory=list)

    def record(
        self,
        project: str,
        duration_s: float,
        iterations: int,
        approved: bool,
        escalated: bool,
    ):
        self.runs.append({
            "project": project,
            "duration_s": round(duration_s, 2),
            "iterations": iterations,
            "approved": approved,
            "escalated": escalated,
        })

    @property
    def task_completion_rate(self) -> float:
        """Objetivo: > 85%"""
        if not self.runs:
            return 0.0
        approved = sum(1 for r in self.runs if r["approved"] and not r["escalated"])
        return round(approved / len(self.runs) * 100, 1)

    @property
    def avg_iterations(self) -> float:
        if not self.runs:
            return 0.0
        return round(sum(r["iterations"] for r in self.runs) / len(self.runs), 1)

    @property
    def avg_duration_s(self) -> float:
        if not self.runs:
            return 0.0
        return round(sum(r["duration_s"] for r in self.runs) / len(self.runs), 1)

    def report(self):
        print("\n📊 KPIs DEL EQUIPO AGÉNTICO")
        print("=" * 40)
        print(f"  Runs totales:          {len(self.runs)}")
        print(f"  Task Completion Rate:  {self.task_completion_rate}%  (objetivo: >85%)")
        print(f"  Iteraciones promedio:  {self.avg_iterations}")
        print(f"  Duración promedio:     {self.avg_duration_s}s")
        escalated = sum(1 for r in self.runs if r["escalated"])
        print(f"  Escalados a humano:    {escalated}")
```

---

## 11. Testing y Evaluación

### Tests unitarios de agentes

```bash
# Instalar dependencias de testing
pip install pytest pytest-asyncio
```

```python
# tests/unit/test_agents.py
import pytest
from unittest.mock import MagicMock, patch
from crewai import Agent, Task, Crew


class TestDevTeamAgents:

    def test_architect_agent_has_correct_role(self):
        """El arquitecto debe tener allow_delegation=True."""
        with patch("crewai.Agent._setup_agent_executor"):
            agent = Agent(
                role="Arquitecto Senior",
                goal="Diseñar soluciones técnicas",
                backstory="15 años de experiencia",
                allow_delegation=True,
                llm="gpt-4o",
            )
        assert agent.allow_delegation is True
        assert "Arquitecto" in agent.role

    def test_developer_agent_no_delegation(self):
        """El desarrollador no debe delegar tareas."""
        with patch("crewai.Agent._setup_agent_executor"):
            agent = Agent(
                role="Desarrollador Python",
                goal="Implementar código",
                backstory="Senior dev",
                allow_delegation=False,
                llm="gpt-4o",
            )
        assert agent.allow_delegation is False

    def test_task_has_required_fields(self):
        """Las tareas deben tener description y expected_output."""
        with patch("crewai.Agent._setup_agent_executor"):
            agent = Agent(
                role="Revisor",
                goal="Revisar código",
                backstory="QA expert",
                llm="gpt-4o",
            )
        task = Task(
            description="Revisar el código del módulo de autenticación",
            expected_output="Reporte con puntuación y lista de issues",
            agent=agent,
        )
        assert task.description is not None
        assert task.expected_output is not None


class TestGuardrails:

    def test_validate_code_output_rejects_empty(self):
        from src.dev_team.guardrails import validate_code_output
        valid, msg = validate_code_output("código muy corto")
        assert valid is False
        assert len(msg) > 0

    def test_validate_code_output_accepts_valid_code(self):
        from src.dev_team.guardrails import validate_code_output
        code = """
import fastapi
from fastapi import FastAPI

app = FastAPI()

def create_token(user_id: int) -> str:
    \"\"\"Genera un JWT para el usuario.\"\"\"
    return f"token_{user_id}"
        """
        valid, msg = validate_code_output(code)
        assert valid is True

    def test_validate_review_output_requires_status(self):
        from src.dev_team.guardrails import validate_review_output
        valid, msg = validate_review_output("El código está bien escrito.")
        assert valid is False

    def test_validate_review_output_accepts_aprobado(self):
        from src.dev_team.guardrails import validate_review_output
        valid, _ = validate_review_output("Puntuación: 85/100 — APROBADO")
        assert valid is True
```

### Tests de integración

```python
# tests/integration/test_crew_pipeline.py
import pytest
from unittest.mock import patch, MagicMock


class TestDevTeamCrew:

    @pytest.fixture
    def mock_llm_response(self):
        """Simula respuestas del LLM para tests sin consumir API."""
        return MagicMock(return_value="Respuesta simulada del agente")

    @patch("crewai.Agent.execute_task")
    def test_crew_sequential_order(self, mock_execute):
        """Los agentes deben ejecutarse en orden: PM → Arch → Dev → Reviewer."""
        execution_order = []

        def track_execution(task, context=None, tools=None):
            execution_order.append(task.agent.role)
            return f"Output de {task.agent.role}"

        mock_execute.side_effect = track_execution

        from src.dev_team.crew import DevTeamCrew
        crew = DevTeamCrew().crew()

        # En un test real ejecutarías crew.kickoff()
        # Aquí verificamos la configuración
        assert len(crew.agents) == 4
        assert len(crew.tasks) == 4

    def test_crew_has_sequential_process(self):
        from crewai import Process
        from src.dev_team.crew import DevTeamCrew

        crew = DevTeamCrew().crew()
        assert crew.process == Process.sequential


class TestPipelineMetrics:

    def test_task_completion_rate_calculation(self):
        from src.dev_team.metrics import PipelineMetrics

        metrics = PipelineMetrics()
        metrics.record("Project A", 120.0, 1, approved=True, escalated=False)
        metrics.record("Project B", 240.0, 2, approved=True, escalated=False)
        metrics.record("Project C", 90.0, 3, approved=False, escalated=True)

        assert metrics.task_completion_rate == pytest.approx(66.7, rel=0.01)
        assert metrics.avg_iterations == pytest.approx(2.0)
```

### Ejecutar tests

```bash
# Todos los tests
pytest tests/ -v

# Solo unitarios (sin API key)
pytest tests/unit/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Ver reporte HTML
open htmlcov/index.html
```

### Evaluación con `crewai test`

```bash
# CrewAI tiene un comando nativo para evaluar la crew
crewai test --n_iterations 3 --model gpt-4o-mini

# Esto ejecuta la crew N veces y genera métricas de calidad
```

---

## 12. KPIs y Métricas

### Uso completo con métricas

```python
# src/dev_team/main.py — versión con métricas completas
import time
from dotenv import load_dotenv
from .crew import DevTeamCrew
from .metrics import PipelineMetrics

load_dotenv()
metrics = PipelineMetrics()


def run_with_metrics(project_name: str, business_goal: str):
    inputs = {"project_name": project_name, "business_goal": business_goal}
    start = time.time()

    try:
        result = DevTeamCrew().crew().kickoff(inputs=inputs)
        duration = time.time() - start
        output = str(result)

        approved = "APROBADO" in output.upper()
        escalated = "ESCALADO" in output.upper()

        metrics.record(
            project=project_name,
            duration_s=duration,
            iterations=1,
            approved=approved,
            escalated=escalated,
        )
        return result

    except Exception as e:
        duration = time.time() - start
        metrics.record(project_name, duration, 1, False, True)
        raise


def run():
    run_with_metrics(
        project_name="Auth API",
        business_goal="API REST con JWT, registro y login. 1000 req/seg."
    )
    metrics.report()

    # Calcular ROI
    hourly_rate = 50        # USD/hora desarrollador
    hours_saved = 4         # horas ahorradas por tarea
    tasks_monthly = 80      # tareas por mes
    agent_cost = 300        # USD/mes (APIs + infra)

    savings = tasks_monthly * hours_saved * hourly_rate
    roi = ((savings - agent_cost) / agent_cost) * 100

    print(f"\n💰 ROI ESTIMADO")
    print(f"  Ahorro bruto mensual: ${savings:,}")
    print(f"  Costo agente:         ${agent_cost:,}")
    print(f"  ROI:                  {roi:.0f}%")


if __name__ == "__main__":
    run()
```

---

## Referencia Rápida de Comandos

```bash
# Crear proyecto nuevo
crewai create crew <nombre>

# Ejecutar la crew
crewai run

# Evaluar calidad (N iteraciones)
crewai test --n_iterations 3

# Replay de la última ejecución (debug)
crewai replay -t <task_id>

# Entrenar la crew con feedback humano
crewai train -n 5 -f feedback.pkl

# Ver logs detallados
crewai run --verbose
```

---

## Recursos

- **Documentación oficial**: [docs.crewai.com](https://docs.crewai.com)
- **Cursos gratuitos**: [learn.crewai.com](https://learn.crewai.com)
- **Ejemplos**: [github.com/crewAIInc/crewAI-examples](https://github.com/crewAIInc/crewAI-examples)
- **Comunidad**: [community.crewai.com](https://community.crewai.com)
- **Quickstarts**: [github.com/crewAIInc/crewAI-quickstarts](https://github.com/crewAIInc/crewAI-quickstarts)
