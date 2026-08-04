"""Behaviour agent package.

Public surface matches the former monolithic ``src.agents.behaviour_agent`` so
existing imports continue to resolve after the split.
"""

from __future__ import annotations

from src.agents.behaviour.agent import BehaviourAgent

__all__ = ["BehaviourAgent"]
