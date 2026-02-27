"""
Tests for the migration workflow system.
"""

import json
from pathlib import Path
from src.state import MigrationState, Phase, MigrationStep
from src.llm_client import LLMClient
from src.agent import MigrationAgent


def test_state_initialization():
    """Test MigrationState initialization."""
    state = MigrationState(
        source_framework="express",
        target_framework="fastapi",
        source_files={"app.js": "const app = require('express')();"},
    )

    assert state.source_framework == "express"
    assert state.target_framework == "fastapi"
    assert state.phase == Phase.ANALYSIS
    assert not state.is_complete()


def test_state_to_dict():
    """Test MigrationState serialization."""
    state = MigrationState(
        source_framework="express",
        target_framework="fastapi",
        source_files={"app.js": "code"},
        migrated_files={"app.py": "migrated code"},
    )

    result = state.to_dict()

    assert result["source_framework"] == "express"
    assert result["target_framework"] == "fastapi"
    assert result["phase"] == "analysis"
    assert len(result["migrated_files"]) == 1


def test_migration_step():
    """Test MigrationStep creation."""
    step = MigrationStep(
        id=1, description="Test step", input_files=["app.js"], output_files=["app.py"]
    )

    assert step.id == 1
    assert step.status == "pending"
    assert len(step.input_files) == 1


def test_step_serialization():
    """Test MigrationStep serialization."""
    step = MigrationStep(
        id=1,
        description="Test step",
        status="completed",
        input_files=["app.js"],
        output_files=["app.py"],
        result="Success",
    )

    result = step.to_dict()

    assert result["id"] == 1
    assert result["status"] == "completed"
    assert result["result"] == "Success"


def test_state_errors():
    """Test error handling in state."""
    state = MigrationState(
        source_framework="test",
        target_framework="test",
        source_files={},
    )

    assert len(state.errors) == 0

    state.add_error("Test error")

    assert len(state.errors) == 1
    assert "Test error" in state.errors


def test_migration_agent_initialization():
    """Test MigrationAgent initialization."""
    agent = MigrationAgent()

    assert agent.llm is not None
    assert isinstance(agent.llm, LLMClient)


def test_phase_transitions():
    """Test state phase transitions."""
    state = MigrationState(
        source_framework="test",
        target_framework="test",
        source_files={},
    )

    assert state.phase == Phase.ANALYSIS

    state.phase = Phase.PLANNING
    assert state.phase == Phase.PLANNING

    state.phase = Phase.EXECUTION
    assert state.phase == Phase.EXECUTION

    state.phase = Phase.VERIFICATION
    assert state.phase == Phase.VERIFICATION

    state.phase = Phase.COMPLETE
    assert state.phase == Phase.COMPLETE
    assert state.is_complete()


def test_migration_state_max_iterations():
    """Test max iterations limit."""
    state = MigrationState(
        source_framework="test",
        target_framework="test",
        source_files={},
        max_iterations=5,
    )

    state.iterations = 5

    assert state.iterations >= state.max_iterations


def run_all_tests():
    """Run all tests."""
    tests = [
        test_state_initialization,
        test_state_to_dict,
        test_migration_step,
        test_step_serialization,
        test_state_errors,
        test_migration_agent_initialization,
        test_phase_transitions,
        test_migration_state_max_iterations,
    ]

    print("\n" + "=" * 60)
    print("🧪 Running Tests")
    print("=" * 60 + "\n")

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
