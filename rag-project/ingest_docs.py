import sys
from pathlib import Path

from pypdf import PdfReader

from app.rag import ingest_text


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json"}


def read_document(path: Path) -> str:
	if path.suffix.lower() == ".pdf":
		return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
	if path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
		return path.read_text(encoding="utf-8")
	raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
	if len(sys.argv) != 2:
		raise SystemExit("Usage: python ingest_docs.py <file-or-directory>")
	target = Path(sys.argv[1])
	paths = [target] if target.is_file() else sorted(
		path for path in target.rglob("*") if path.is_file()
	)
	if not paths:
		raise SystemExit(f"No documents found at {target}")
	for path in paths:
		try:
			count = ingest_text(str(path), read_document(path))
			print(f"Indexed {path}: {count} chunks")
		except ValueError as exc:
			print(f"Skipped {path}: {exc}")


if __name__ == "__main__":
	main()
