# verify_setup.py
from __future__ import annotations

import importlib
from importlib import metadata


PACKAGES: list[tuple[str, str]] = [
    # (import_name, pypi_distribution_name)
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("pinecone", "pinecone"),
    ("boto3", "boto3"),
    ("llama_index", "llama-index"),
    ("mcp", "mcp"),
]


def get_version(import_name: str, dist_name: str) -> str:
    # Confirm import works
    importlib.import_module(import_name)

    # Prefer dist metadata (most reliable)
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "unknown (dist metadata not found)"


def main() -> None:
    print("Checking installations...")
    print(f"Python: {importlib.import_module('sys').executable}\n")

    for import_name, dist_name in PACKAGES:
        try:
            ver = get_version(import_name, dist_name)
            print(f"OK {dist_name} (import: {import_name}) - version {ver}")
        except Exception as e:
            print(f"FAIL {dist_name} (import: {import_name}) - {e}")

    print("\nSetup verification complete!")


if __name__ == "__main__":
    main()