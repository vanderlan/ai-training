"""
CLI utility for running migrations from the command line.
"""

import json
import argparse
from pathlib import Path
from src.agent import MigrationAgent
from src.llm_client import LLMClient
from src.state import MigrationState


def load_files_from_directory(directory: str) -> dict[str, str]:
    """Load source files from a directory."""
    files = {}
    path = Path(directory)

    for file in path.rglob("*"):
        if file.is_file() and not any(part.startswith(".") for part in file.parts):
            try:
                files[str(file.relative_to(path))] = file.read_text()
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")

    return files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Migration Workflow System CLI")

    parser.add_argument("source_framework", help="Source framework (e.g., 'express', 'django')")
    parser.add_argument("target_framework", help="Target framework (e.g., 'fastapi', 'nest')")
    parser.add_argument("files_or_directory", help="Source files or directory containing files")
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory for migrated files",
        default="./migrated",
    )
    parser.add_argument(
        "--json",
        "-j",
        help="Output results as JSON",
        action="store_true",
    )

    args = parser.parse_args()

    # Load source files
    if Path(args.files_or_directory).is_dir():
        files = load_files_from_directory(args.files_or_directory)
    else:
        # Single file
        path = Path(args.files_or_directory)
        files = {path.name: path.read_text()}

    if not files:
        print("❌ No files found")
        return

    print(f"\n{'='*60}")
    print(f"🚀 Migration Workflow System - CLI")
    print(f"{'='*60}")
    print(f"Source Framework: {args.source_framework}")
    print(f"Target Framework: {args.target_framework}")
    print(f"Files to migrate: {len(files)}")
    print(f"{'='*60}\n")

    # Create agent and run migration
    llm_client = LLMClient()
    agent = MigrationAgent(llm_client)

    state = MigrationState(
        source_framework=args.source_framework,
        target_framework=args.target_framework,
        source_files=files,
    )

    state = agent.run(state)

    # Output results
    if args.json:
        result = state.to_dict()
        print(json.dumps(result, indent=2))
    else:
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Migration Summary")
        print(f"{'='*60}")
        print(f"Status: {'✅ Success' if len(state.errors) == 0 else '❌ Failed'}")
        print(f"Phase: {state.phase.value}")
        print(f"Files migrated: {len(state.migrated_files)}")
        print(f"Steps completed: {sum(1 for s in state.plan if s.status == 'completed')}/{len(state.plan)}")

        if state.errors:
            print(f"\nErrors:")
            for error in state.errors:
                print(f"  ❌ {error}")

        # Save migrated files
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        for filename, content in state.migrated_files.items():
            file_path = output_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            print(f"Saved: {file_path}")

        print(f"\n✓ Results saved to {output_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
