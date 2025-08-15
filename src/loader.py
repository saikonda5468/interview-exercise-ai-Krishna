from pathlib import Path

def load_docs(data_dir_path: str) -> dict[str, str]:
    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    docs: dict[str, str] = {}
    for file in data_dir.glob("*.txt"):
        content = file.read_text(encoding="utf-8").strip()
        if content:
            title = file.stem.replace("_", " ").title()
            docs[title] = content
    if not docs:
        raise ValueError(f"No valid .txt files found in {data_dir}")
    return docs