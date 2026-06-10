"""LLaVA Inference Service: envoltorio FastAPI sobre llama.cpp."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from llama_cpp import Llama
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    model_path: str = Field("/app/models/ggml-model-q4_k.gguf", alias="LLAVA_MODEL_PATH")
    n_threads: int = Field(4, alias="LLAVA_N_THREADS")
    n_ctx: int = Field(1024, alias="LLAVA_N_CTX")
    n_batch: int = Field(64, alias="LLAVA_N_BATCH")


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llava-service")


app = FastAPI(
    title="LLaVA Inference Service",
    version="1.0.0",
    description="Servicio de inferencia local usando LLaVA via llama.cpp",
)


try:
    logger.info("Cargando modelo LLaVA desde %s", settings.model_path)
    llm = Llama(
        model_path=settings.model_path,
        n_threads=settings.n_threads,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        use_mmap=True,
        use_mlock=False,
    )
    logger.info("LLaVA cargado correctamente (n_ctx=%d, n_threads=%d)", settings.n_ctx, settings.n_threads)
except Exception as e:
    logger.exception("Error fatal cargando LLaVA")
    raise RuntimeError(f"Error al cargar Llava desde {settings.model_path}: {e}") from e


class InferRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada para que el modelo genere una respuesta.")
    max_tokens: int = Field(256, ge=1, le=2048, description="Numero maximo de tokens a generar.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Aleatoriedad de la generacion.")


class InferResponse(BaseModel):
    text: str = Field(..., description="Texto generado por el modelo")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio.")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    try:
        result = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        choices = result.get("choices") if isinstance(result, dict) else None
        if choices and isinstance(choices, list) and len(choices) > 0:
            generated = choices[0].get("text", "")
        else:
            generated = ""
        return InferResponse(text=generated.strip())
    except Exception as e:
        logger.exception("Fallo en inferencia LLaVA")
        raise HTTPException(status_code=500, detail=str(e)) from e
