"""Supervisor agent that coordinates worker agents."""
import json
import logging
from typing import Callable, Dict, Generator, List, Optional

from src.agents import ResearcherAgent, WriterAgent, ReviewerAgent
from src.llm_client import LLMClient
from src.cost_tracker import cost_tracker
from src.security import check_prompt_injection, validate_llm_output

logger = logging.getLogger("multi_agent.supervisor")

SUPERVISOR_PROMPT = """You are a supervisor managing a team of specialized agents.

Available agents:
- Researcher: Finds and summarizes information on any topic
- Writer: Creates polished content from research material
- Reviewer: Reviews content for quality and accuracy

Your job:
1. Analyze the incoming task
2. Decide which agent(s) to use and in what order
3. Coordinate their work step by step
4. Synthesize the final output

IMPORTANT: You must delegate work ONE STEP AT A TIME.

For each step, output EXACTLY in this format (nothing else):
DELEGATE: <agent_name>
TASK: <specific task description for that agent>

When all work is done and you have a satisfactory result, output:
FINAL: <the complete, synthesized final output>

Typical workflow:
1. DELEGATE to Researcher to gather information
2. DELEGATE to Writer to produce polished content from the research
3. Optionally DELEGATE to Reviewer for quality check
4. Output FINAL with the best version of the content"""


class SupervisorAgent:
    """Supervisor that coordinates worker agents."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.workers = {
            "Researcher": ResearcherAgent(llm_client),
            "Writer": WriterAgent(llm_client),
            "Reviewer": ReviewerAgent(llm_client),
        }

    def run(self, task: str, max_iterations: int = 5) -> Dict:
        """
        Run the multi-agent workflow.
        Returns dict with result, steps taken, and execution log.
        """
        results: Dict[str, str] = {}
        steps_log: List[Dict] = []

        messages = [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ]

        for i in range(max_iterations):
            # Budget check
            if not cost_tracker.check_budget():
                logger.warning("Budget exhausted at iteration %d", i)
                return {
                    "result": self._force_final(results),
                    "steps": steps_log,
                    "steps_taken": len(steps_log),
                    "note": "Budget limit reached — returning best available result.",
                }

            # Get supervisor decision
            response = self.llm.chat(messages)
            response = validate_llm_output(response)
            messages.append({"role": "assistant", "content": response})

            logger.info("Supervisor iteration %d: %s", i + 1, response[:200])

            # Check for final output
            if "FINAL:" in response:
                final = response.split("FINAL:", 1)[-1].strip()
                steps_log.append({
                    "iteration": i + 1,
                    "action": "final",
                    "agent": "Supervisor",
                    "detail": "Synthesized final output",
                })
                return {
                    "result": final,
                    "steps": steps_log,
                    "steps_taken": len(steps_log),
                }

            # Parse delegation
            if "DELEGATE:" in response and "TASK:" in response:
                agent_name = response.split("DELEGATE:", 1)[-1].split("TASK:")[0].strip()
                agent_task = response.split("TASK:", 1)[-1].strip()

                if agent_name in self.workers:
                    # Execute worker
                    context = self._get_context(results)
                    worker_result = self.workers[agent_name].execute(agent_task, context)

                    # Store result
                    key = f"{agent_name}_{i}"
                    results[key] = worker_result

                    steps_log.append({
                        "iteration": i + 1,
                        "action": "delegate",
                        "agent": agent_name,
                        "task": agent_task[:200],
                        "result_preview": worker_result[:300],
                    })

                    # Feed result back to supervisor
                    messages.append({
                        "role": "user",
                        "content": f"Result from {agent_name}:\n{worker_result}",
                    })
                else:
                    logger.warning("Unknown agent '%s' — skipping", agent_name)
                    messages.append({
                        "role": "user",
                        "content": f"Error: No agent named '{agent_name}'. "
                                   f"Available: {', '.join(self.workers.keys())}",
                    })

        # Max iterations reached
        return {
            "result": self._force_final(results),
            "steps": steps_log,
            "steps_taken": len(steps_log),
            "note": "Max iterations reached — returning best available result.",
        }

    def _get_context(self, results: Dict[str, str]) -> str:
        """Build context from previous worker results."""
        if not results:
            return ""
        parts = []
        for key, value in results.items():
            parts.append(f"--- {key} ---\n{value}")
        return "\n\n".join(parts)

    def _force_final(self, results: Dict[str, str]) -> str:
        """Return best available result if max iterations reached."""
        if results:
            # Prefer writer output
            writer_results = [v for k, v in results.items() if "Writer" in k]
            if writer_results:
                return writer_results[-1]
            # Fall back to last result
            return list(results.values())[-1]
        return "Unable to complete task."

    # ------------------------------------------------------------------
    # Streaming variant — yields SSE events during execution
    # ------------------------------------------------------------------
    def run_streaming(self, task: str, max_iterations: int = 5) -> Generator[str, None, None]:
        """
        Run the multi-agent workflow, yielding Server-Sent Events as JSON lines.
        Event types: thinking, delegating, working, result, final, error, done
        """
        results: Dict[str, str] = {}
        steps_log: List[Dict] = []

        def _evt(event_type: str, data: dict) -> str:
            payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
            return f"data: {payload}\n\n"

        messages = [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ]

        yield _evt("start", {"task": task, "max_iterations": max_iterations})

        for i in range(max_iterations):
            # Budget check
            if not cost_tracker.check_budget():
                yield _evt("error", {"message": "Budget limit reached"})
                break

            # Supervisor is thinking
            yield _evt("thinking", {
                "agent": "Supervisor",
                "iteration": i + 1,
                "message": "Analyzing task and deciding next step…",
            })

            response = self.llm.chat(messages)
            response = validate_llm_output(response)
            messages.append({"role": "assistant", "content": response})

            # Final output
            if "FINAL:" in response:
                final = response.split("FINAL:", 1)[-1].strip()
                steps_log.append({
                    "iteration": i + 1,
                    "action": "final",
                    "agent": "Supervisor",
                    "detail": "Synthesized final output",
                })
                yield _evt("final", {
                    "result": final,
                    "steps": steps_log,
                    "steps_taken": len(steps_log),
                })
                yield _evt("done", {})
                return

            # Delegation
            if "DELEGATE:" in response and "TASK:" in response:
                agent_name = response.split("DELEGATE:", 1)[-1].split("TASK:")[0].strip()
                agent_task = response.split("TASK:", 1)[-1].strip()

                if agent_name in self.workers:
                    yield _evt("delegating", {
                        "from": "Supervisor",
                        "to": agent_name,
                        "task": agent_task[:200],
                        "iteration": i + 1,
                    })

                    yield _evt("working", {
                        "agent": agent_name,
                        "iteration": i + 1,
                        "message": f"{agent_name} is working…",
                    })

                    context = self._get_context(results)
                    worker_result = self.workers[agent_name].execute(agent_task, context)

                    key = f"{agent_name}_{i}"
                    results[key] = worker_result

                    step_entry = {
                        "iteration": i + 1,
                        "action": "delegate",
                        "agent": agent_name,
                        "task": agent_task[:200],
                        "result_preview": worker_result[:300],
                    }
                    steps_log.append(step_entry)

                    yield _evt("result", {
                        "agent": agent_name,
                        "iteration": i + 1,
                        "preview": worker_result[:500],
                    })

                    messages.append({
                        "role": "user",
                        "content": f"Result from {agent_name}:\n{worker_result}",
                    })
                else:
                    yield _evt("error", {
                        "message": f"Unknown agent '{agent_name}'",
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Error: No agent named '{agent_name}'. "
                                   f"Available: {', '.join(self.workers.keys())}",
                    })

        # Max iterations reached
        final_text = self._force_final(results)
        yield _evt("final", {
            "result": final_text,
            "steps": steps_log,
            "steps_taken": len(steps_log),
            "note": "Max iterations reached — returning best available result.",
        })
        yield _evt("done", {})
