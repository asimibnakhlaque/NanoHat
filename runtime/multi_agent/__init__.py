"""
runtime/multi_agent/
Specialized 4-agent pipeline for SmolLM2 on Fedora Linux:
- RouterAgent (smollm2:135m): Binary intent routing
- PlannerAgent (smollm2:360m): Structured task planning and failure rediagnosis
- ExecutorAgent (nanohat2:360m): Rolling conversation tool execution
- ChitchatAgent (smollm2:360m): Conversational responses and tool synthesis
- NanoHatOrchestrator: Two-layer memory management and pipeline coordinator
"""

from runtime.multi_agent.base_agent import BaseAgent
from runtime.multi_agent.router import RouterAgent
from runtime.multi_agent.planner import PlannerAgent
from runtime.multi_agent.executor import ExecutorAgent
from runtime.multi_agent.chitchat import ChitchatAgent
from runtime.multi_agent.orchestrator import NanoHatOrchestrator

__all__ = [
    "BaseAgent",
    "RouterAgent",
    "PlannerAgent",
    "ExecutorAgent",
    "ChitchatAgent",
    "NanoHatOrchestrator",
]
