"""FastAPI 웹 서버. 화면(정적 파일)과 추론 API를 한 서버에서 함께 제공한다."""
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import MAX_UPLOAD_BYTES, InputError, Predictor

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="급제동 탐지 서비스", version="1.0.0")
predictor = Predictor()          # 서버가 뜰 때 학습한 모델 파일을 한 번 읽는다


@app.get("/api/health")
def health():
    return {"status": "ok", "model": predictor.meta["model_name"]}


@app.get("/api/model")
def model_info():
    return predictor.model_info()


@app.get("/api/samples")
def samples():
    return predictor.list_samples()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """CAN JSON 파일을 받아 급제동 확률을 돌려준다."""
    name = (file.filename or "").lower()
    if not name.endswith(".json"):
        raise HTTPException(status_code=400, detail="JSON 파일만 올릴 수 있습니다. (.json)")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        return predictor.analyze(raw)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-sample/{sample_id}")
def predict_sample(sample_id: str):
    """저장소에 들어 있는 예시 파일로 추론한다."""
    try:
        return predictor.analyze_sample(sample_id)
    except InputError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
