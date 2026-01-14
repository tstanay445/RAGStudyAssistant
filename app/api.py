from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import run_ingestion
from app.rag import answer_question, reload_index

app = FastAPI()


class AskRequest(BaseModel):
    question: str
    session_id : str


@app.on_event("startup")
def startup():
    reload_index()


from fastapi import UploadFile, File
import tempfile
import shutil
import os

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    tmpdir = tempfile.mkdtemp()

    for f in files:
        dest = os.path.join(tmpdir, f.filename)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

    run_ingestion(tmpdir)
    reload_index()

    return {"status": "ok"}



@app.post("/ask")
def ask(req: AskRequest):
    answer = answer_question(req.question,req.session_id)
    return {"answer": answer}
