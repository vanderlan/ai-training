"""
State Management for Migration Workflow System

Defines all data structures and state transitions for the migration agent.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class Phase(str, Enum):
    """Migration workflow phases."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETE = "complete"


@dataclass
class MigrationStep:
    """Represents a single migration step."""
    id: int
    description: str
    status: str = "pending"
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class MigrationState:
    """Represents the complete state of a migration."""
    source_framework: str
    target_framework: str
    source_files: Dict[str, str]
    phase: Phase = Phase.ANALYSIS
    analysis: Optional[Dict[str, Any]] = None
    plan: List[MigrationStep] = field(default_factory=list)
    current_step: int = 0
    migrated_files: Dict[str, str] = field(default_factory=dict)
    verification_result: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 20

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def is_complete(self) -> bool:
        """Check if migration is complete."""
        return self.phase == Phase.COMPLETE or len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_framework": self.source_framework,
            "target_framework": self.target_framework,
            "phase": self.phase.value,
            "current_step": self.current_step,
            "analysis": self.analysis,
            "plan": [step.to_dict() for step in self.plan],
            "migrated_files": self.migrated_files,
            "verification_result": self.verification_result,
            "errors": self.errors,
            "iterations": self.iterations,
        }
