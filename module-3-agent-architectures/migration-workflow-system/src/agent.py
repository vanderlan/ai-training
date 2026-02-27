"""
Migration Agent Implementation.

Core agent that orchestrates the migration workflow through all phases.
"""

from typing import Optional
from src.state import MigrationState, Phase, MigrationStep
from src.llm_client import LLMClient
from src.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    EXECUTION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)


class MigrationAgent:
    """Agent that orchestrates code migrations through the Observe → Think → Act cycle."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the migration agent."""
        self.llm = llm_client or LLMClient()

    def run(self, state: MigrationState) -> MigrationState:
        """Run the agent loop until completion or error."""
        while state.phase != Phase.COMPLETE:
            state = self._step(state)
            state.iterations += 1

            if state.iterations >= state.max_iterations:
                state.add_error(f"Max iterations ({state.max_iterations}) reached")
                state.phase = Phase.COMPLETE
                break

        return state

    def _step(self, state: MigrationState) -> MigrationState:
        """Single step of the agent - observe, think, act."""
        if state.phase == Phase.ANALYSIS:
            return self._analyze(state)
        elif state.phase == Phase.PLANNING:
            return self._plan(state)
        elif state.phase == Phase.EXECUTION:
            return self._execute(state)
        elif state.phase == Phase.VERIFICATION:
            return self._verify(state)
        else:
            return state

    # PHASE 1: ANALYSIS - Observe the source code
    def _analyze(self, state: MigrationState) -> MigrationState:
        """Phase 1: Analyze the source files and framework patterns."""
        try:
            print(f"🔍 ANALYSIS: Analyzing {len(state.source_files)} files...")

            # Think: Use LLM to analyze
            analysis = self.llm.generate_analysis(ANALYSIS_SYSTEM_PROMPT, state.source_files)

            state.analysis = analysis
            state.phase = Phase.PLANNING

            print(f"✓ Analysis complete. Found {len(analysis.get('key_components', {}))} components.")
            return state

        except Exception as e:
            state.add_error(f"Analysis failed: {str(e)}")
            state.phase = Phase.COMPLETE
            return state

    # PHASE 2: PLANNING - Think about what to do
    def _plan(self, state: MigrationState) -> MigrationState:
        """Phase 2: Create a step-by-step migration plan."""
        try:
            print(f"📋 PLANNING: Creating migration plan from {state.source_framework} to {state.target_framework}...")

            # Think: Use LLM to create plan
            plan_response = self.llm.generate_plan(
                PLANNING_SYSTEM_PROMPT, state.source_framework, state.target_framework, state.analysis or {}
            )

            # Convert plan response to MigrationStep objects
            steps = []
            for step_data in plan_response.get("steps", []):
                step = MigrationStep(
                    id=step_data.get("id", len(steps) + 1),
                    description=step_data.get("title", "Untitled step"),
                    status="pending",
                    input_files=step_data.get("input_files", []),
                    output_files=step_data.get("output_files", []),
                )
                steps.append(step)

            state.plan = steps
            state.phase = Phase.EXECUTION

            print(f"✓ Plan created with {len(steps)} steps.")
            return state

        except Exception as e:
            state.add_error(f"Planning failed: {str(e)}")
            state.phase = Phase.COMPLETE
            return state

    # PHASE 3: EXECUTION - Act on the plan
    def _execute(self, state: MigrationState) -> MigrationState:
        """Phase 3: Execute migration steps."""
        try:
            # Check if all steps are complete
            if state.current_step >= len(state.plan):
                print(f"✓ All {len(state.plan)} steps executed. Moving to verification...")
                state.phase = Phase.VERIFICATION
                return state

            step = state.plan[state.current_step]
            print(f"⚙️  EXECUTING: Step {step.id + 1}/{len(state.plan)} - {step.description}...")

            # Prepare input files for this step
            input_files = {}
            for file_name in step.input_files:
                if file_name in state.source_files:
                    input_files[file_name] = state.source_files[file_name]
                elif file_name in state.migrated_files:
                    input_files[file_name] = state.migrated_files[file_name]

            if not input_files:
                input_files = state.source_files  # Use all source files if none specified

            # Act: Use LLM to execute step
            step_result = self.llm.execute_step(
                EXECUTION_SYSTEM_PROMPT,
                step.description,
                input_files,
                state.target_framework,
            )

            # Process results
            if step_result.get("status") == "completed":
                step.status = "completed"

                # Store generated files
                for filename, content in step_result.get("generated_files", {}).items():
                    state.migrated_files[filename] = content

                step.result = step_result.get("summary", "")
            else:
                step.status = "failed"
                step.error = step_result.get("summary", "Step failed")
                state.add_error(f"Step {step.id} failed: {step.error}")

            # Log any issues
            for issue in step_result.get("issues_encountered", []):
                print(f"  ⚠️  {issue}")

            state.current_step += 1
            return state

        except Exception as e:
            if state.current_step < len(state.plan):
                step = state.plan[state.current_step]
                step.status = "failed"
                step.error = str(e)
                state.add_error(f"Step {step.id} execution error: {str(e)}")
            else:
                state.add_error(f"Execution error: {str(e)}")
            state.current_step += 1
            return state

    # PHASE 4: VERIFICATION - Verify the migration worked
    def _verify(self, state: MigrationState) -> MigrationState:
        """Phase 4: Verify the migrated code."""
        try:
            print(f"✅ VERIFICATION: Verifying migrated code...")

            # Verify all migrated files
            verification = self.llm.verify_migration(
                VERIFICATION_SYSTEM_PROMPT, state.migrated_files or {}, state.target_framework
            )

            state.verification_result = verification

            # Check verification status
            status = verification.get("verification_status", "passed")
            if status == "failed":
                state.add_error("Verification failed - see details in verification_result")

            state.phase = Phase.COMPLETE

            print(f"✓ Verification complete. Status: {status}")
            return state

        except Exception as e:
            state.add_error(f"Verification failed: {str(e)}")
            state.phase = Phase.COMPLETE
            return state
