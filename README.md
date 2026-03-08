# 🤖 Agentic Dev Team — CrewAI

Sistema multi-agente para equipos de desarrollo de software, construido con Python, LangChain y LangGraph.

---

## ¿Qué es esto?

Un framework que convierte la IA de herramienta de consulta a **miembro activo del equipo**: agentes especializados que planifican, escriben código, lo revisan y se autocorrigen — con observabilidad completa y control de versiones de prompts.

```
Arquitecto ──▶ Desarrollador ──▶ Revisor
    │                               │
    └───────── reintento ───────────┘  (máx. 3 iteraciones)
```

---

## Inicio rápido

```bash
# 1. Clonar e instalar
git clone https://github.com/tu-org/agentic-dev-team
cd agentic-dev-team
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configurar credenciales
cp .env.example .env
# Editar .env con tus API keys

# 3. Ejecutar ejemplo
python main.py
```

### `.env` mínimo

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-dev-team
```

---

## Estructura

```
agentic_team/
├── agents/              # Arquitecto, Desarrollador, Revisor, PM
├── orchestration/       # Secuencial, concurrente, LangGraph
├── memory/              # Buffer (corto plazo) + ChromaDB (largo plazo)
├── tools/               # Git, ejecución de código, APIs externas
├── observability/       # Trazado asíncrono + métricas KPI
├── governance/          # Registro de prompts con versionado
├── evaluation/          # Golden dataset + LLM-as-a-Judge
└── tests/               # Unit · Integration · Evaluation
```

---

## Agentes incluidos

| Agente | Rol | No hace |
|---|---|---|
| **Arquitecto** | Analiza requisitos y genera el plan técnico | Escribir código directamente |
| **Desarrollador** | Implementa, ejecuta y valida el código | Tomar decisiones de diseño |
| **Revisor** | Evalúa calidad, seguridad y estilo | Aprobar sin criterios objetivos |
| **PM** | Prioriza backlog y alinea con negocio | Definir implementación técnica |

---

## Patrones de orquestación

```python
from orchestration.sequential import SequentialOrchestrator
from orchestration.concurrent import ConcurrentOrchestrator

# Secuencial: Arquitecto → Desarrollador → Revisor
pipeline = SequentialOrchestrator(agents=[architect, developer, reviewer])
result = pipeline.run("Implementar endpoint de autenticación JWT")

# Concurrente: múltiples perspectivas en paralelo
brainstorm = ConcurrentOrchestrator(agents=[dev_a, dev_b, dev_c], aggregation="vote")
result = brainstorm.run("¿Cuál es la mejor estrategia de caché?")
```

---

## Testing

```bash
# Suite completa
pytest tests/ -v

# Solo unitarios (sin LLM, rápido)
pytest tests/unit/ -v

# Con cobertura
pytest tests/ --cov=agents --cov=orchestration --cov-report=html
```

---

## KPIs monitoreados

| Métrica | Objetivo |
|---|---|
| Task Completion Rate | > 85% |
| First Contact Resolution | > 70% |
| Tasa de Contención | Maximizar |
| Latencia P99 | Minimizar |

---

## Dependencias principales

- `langgraph` — orquestación con grafos de estado
- `langchain` + `langchain-openai` — agentes y herramientas
- `chromadb` — memoria vectorial persistente
- `langsmith` — trazado y observabilidad
- `pytest` + `pytest-asyncio` — testing

---

## Documentación

Consulta la **[Guía de Implementación completa](guia-arquitectura-agentica-python.md)** para detalles de cada componente, incluyendo código de todos los módulos, patrones de manejo de errores y configuración del ADLC.

---

## Licencia

MIT
