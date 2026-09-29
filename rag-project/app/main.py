import io
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .rag import answer_query, ingest_text, stream_answer_query

app = FastAPI(title="Local RAG API")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf"}


def extract_text_from_upload(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in SUPPORTED_EXTENSIONS:
        return content.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file format: {ext}")


class IngestRequest(BaseModel):
    source: str
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=2, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        file_bytes = await file.read()
        extracted_text = extract_text_from_upload(file.filename, file_bytes)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400, detail="The uploaded file contains no readable text."
            )

        chunk_count = ingest_text(file.filename, extracted_text)
        return {
            "source": file.filename,
            "chunks": chunk_count,
            "message": f"Successfully indexed '{file.filename}' into {chunk_count} chunks.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/ingest")
def ingest(request: IngestRequest):
    try:
        count = ingest_text(request.source, request.text)
        return {"source": request.source, "chunks": count}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query")
def query(request: QueryRequest):
    try:
        return answer_query(request.question, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    try:
        generator_fn, sources = stream_answer_query(request.question, request.top_k)

        def sse():
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            for token in generator_fn():
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Mount static frontend (must remain after API route definitions)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")