import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.predictor import ChurnPredictor
from src.api.schemas import CustomerInput, PredictionOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Predictor API",
    description="Previsão de churn para clientes de telecomunicações",
    version="1.0.0",
)

predictor = ChurnPredictor()


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    prob, prediction = predictor.predict(customer)
    logger.info("prediction_made", churn_probability=prob, churn_prediction=prediction)
    return PredictionOutput(churn_probability=prob, churn_prediction=prediction)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
