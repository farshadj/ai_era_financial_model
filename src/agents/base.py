"""
Base agent class for the agent-based economic simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentState:
    """Mutable state that changes each step."""
    wealth: float = 0.0
    income: float = 0.0
    employed: bool = True
    consumption: float = 0.0
    savings_rate: float = 0.15
    sentiment: float = 0.5   # 0 = fearful, 1 = greedy


class BaseAgent:
    """
    Base class for all agents in the ABM.

    Each agent has:
    - An immutable ID and type
    - Mutable state
    - A step() method called each simulation tick
    """

    def __init__(self, agent_id: int, agent_type: str, **kwargs):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(**{k: v for k, v in kwargs.items()
                                    if k in AgentState.__dataclass_fields__})
        self.properties: Dict[str, Any] = {k: v for k, v in kwargs.items()
                                             if k not in AgentState.__dataclass_fields__}

    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one simulation step. Override in subclasses.

        Args:
            environment: Global state visible to all agents.

        Returns:
            Dictionary of actions/outputs for this step.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.agent_type}(id={self.agent_id})"
