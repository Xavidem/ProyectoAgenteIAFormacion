from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama
import os

# Ruta del modelo
MODEL_PATH = os.getenv("LLAVA_MODEL_PATH", "/app/models/ggml-model-q4_k.gguf")

# Llamamos a FastAPI
app = FastAPI(
   title = "LLaVA Inference Service",
   version = "1.0.0",
   description = "Servicio de inferencia local usando LLaVA via llama.cpp"
)

# Cargamos el modelo en memoria
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_threads=int(os.getenv("LLAVA_N_THREADS", "4")),
        n_ctx = int(os.getenv("LLAVA_N_CTX", "2048")),
        n_batch = int(os.getenv("LLAVA_N_BATCH", "128"))
    )
except Exception as e:
    raise RuntimeError("Error al cargar Llava desde {MODEL_PATH}: {e}")


# Escribimos los Pydantic models
class InferRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada para que el modelo genere una respuesta.")
    max_tokens: int = Field(256, description="Numero maximo de tokens a generar en la respuesta.")
    temperature: float = Field(0.7, description="Control de la aleatoriedad de la generacion de texto: 0.0 - 1.0")

class InferResponse(BaseModel):
    text: str = Field(..., description="Texto generado por el modelo")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio.")


# Endpoint para la salud del servicio
"""
Comprueba que el servicio esta funcionando correctamente
"""
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


# Endpoint para inferencia
"""
Generar una respuesta a partir de un texto dado un prompt de usuario. Usamos llama.cpp para ejecutar LLaVA
"""
@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest):
    try:
        result = llm(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        choices = result.get("choices") if isinstance(result, dict) else None
        if choices and isinstance(choices, list) and len(choices) > 0:
            generated = choices[0].get("text", "")
        else:
            generated = ""
        return InferResponse(text=generated.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
