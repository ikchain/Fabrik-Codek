#!/usr/bin/env python3
"""Install fabrik-logger in a project.

This script sets up the fabrik-logger in any project by:
1. Adding fabrik-codek to the Python path
2. Creating a simple import wrapper

Usage:
    # From any project directory:
    python scripts/install_logger.py

    # Or with project name:
    python scripts/install_logger.py --project myproject
"""

import argparse
import sys
from pathlib import Path

FABRIK_PATH = Path(__file__).resolve().parent.parent
DATALAKE_PATH = FABRIK_PATH / "data"


def create_logger_wrapper(project_root: Path, project_name: str) -> None:
    """Create a simple logger wrapper in the project."""
    utils_dir = project_root / "utils"
    utils_dir.mkdir(exist_ok=True)

    wrapper_content = f'''"""Fabrik Logger wrapper for {project_name}.

This imports the centralized logger from fabrik-codek.
All data goes to: {DATALAKE_PATH}

Usage:
    from utils.logger import get_logger

    logger = get_logger()
    logger.log_code_change(...)
"""

import sys
from pathlib import Path

# Add fabrik-codek to path
fabrik_path = Path("{FABRIK_PATH}/src")
if str(fabrik_path) not in sys.path:
    sys.path.insert(0, str(fabrik_path))

from flywheel.logger import FabrikLogger, get_logger as _get_logger, reset_logger

PROJECT_NAME = "{project_name}"

def get_logger() -> FabrikLogger:
    """Get logger instance for this project."""
    return _get_logger(project_name=PROJECT_NAME)

__all__ = ["get_logger", "FabrikLogger", "reset_logger"]
'''

    wrapper_path = utils_dir / "logger.py"
    wrapper_path.write_text(wrapper_content)
    print(f"  Created: {wrapper_path}")


def main():
    parser = argparse.ArgumentParser(description="Install fabrik-logger in a project")
    parser.add_argument("--project", "-p", help="Project name (auto-detected if not specified)")
    parser.add_argument("--path", help="Project path (defaults to cwd)")
    args = parser.parse_args()

    project_root = Path(args.path) if args.path else Path.cwd()
    project_name = args.project or project_root.name

    print(f"Installing fabrik-logger for: {project_name}")
    print(f"  Project root: {project_root}")
    print(f"  Datalake: {DATALAKE_PATH}")
    print()

    # Create wrapper
    create_logger_wrapper(project_root, project_name)

    print()
    print("Done! Usage:")
    print("  from utils.logger import get_logger")
    print("  logger = get_logger()")
    print("  logger.log_code_change(...)")


if __name__ == "__main__":
    main()
