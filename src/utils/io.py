"""
I/O helper utilities for reading and writing files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def read_text_file(filepath: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(filepath: str, content: str) -> None:
    """Write content to a UTF-8 text file, creating parent dirs if needed."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def read_json(filepath: str) -> Dict[str, Any]:
    """Read and parse a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: str, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def list_files(directory: str, extensions: tuple = (".pdf", ".docx", ".txt")) -> list:
    """List files in a directory matching given extensions."""
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f)[1].lower() in extensions
    ]
