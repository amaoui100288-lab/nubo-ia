import os
import requests
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from replit import db # Usamos Replit DB para memoria gratis

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

class Mensaje(BaseModel):
    user: str
    message: str

def generar_imagen(prompt):
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}"}
    data = {
        "version": "black-forest-labs/flux-dev",
        "input": {"prompt": f"masterpiece, photorealistic, 8k, detailed, {prompt}"}
    }
    r = requests.post("https://api.replicate.com/v1/predictions", json=data, headers=headers)
    pred = r.json()

    while pred["status"] not in ["succeeded", "failed"]:
        time.sleep(2)
        pred = requests.get(pred["urls"]["get"], headers=headers).json()

    return pred["output"][0] if pred["status"] == "succeeded" else None

def preguntar_ollama(prompt, memoria, conocimiento):
    full_prompt = f"""
    Eres Nubo IA, un asistente inteligente y amigable.
    Memoria de conversación: {memoria}
    Base de conocimiento: {conocimiento}

    Usuario: {prompt}
    Nubo:
    """
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False})
    return r.json()["response"]

@app.post("/chat")
async def chat(req: Mensaje):
    memoria = db.get(f"mem_{req.user}", "")
    conocimiento = db.get("pdfs", "")[:4000] # Base de conocimiento

    # Si pide imagen
    if "imagen" in req.message.lower() or "genera" in req.message.lower() or "dibuja" in req.message.lower():
        url_img = generar_imagen(req.message)
        return {"reply": "Listo jefe, aquí tienes tu imagen 🔥", "image": url_img}

    # Si no, responde con texto
    respuesta = preguntar_ollama(req.message, memoria, conocimiento)

    # Guardar memoria
    nueva_memoria = memoria + f"\nUsuario: {req.message}\nNubo: {respuesta}"
    db[f"mem_{req.user}"] = nueva_memoria[-2000:] # Guardamos ultimos 2000 chars

    return {"reply": respuesta, "image": None}

@app.post("/upload_pdf")
async def upload_texto(texto: str):
    db["pdfs"] = db.get("pdfs", "") + "\n" + texto
    return {"status": "ok", "msg": "Conocimiento añadido a Nubo"}
