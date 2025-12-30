# app/main.py

import os
from dotenv import load_dotenv  # 👈 add this

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth_routes import router as auth_router
from app.api.tables_routes import router as tables_router
from routes import es_test
from app.api.documents_routes import router as documents_router
from app.api.chat_routes import router as chat_router
from app.api.docs_analytics_routes import router as docs_analytics_router

# 🔹 Load .env BEFORE creating app / routers
load_dotenv()  # 👈 this is what makes OPENAI_API_KEY visible

app = FastAPI(title="Agentic AI API")

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://agentic-ia-frontend.vercel.app",
    "https://agentic-ia-frontend-r27k8srw6-yessines-projects-009bbb37.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(auth_router)
app.include_router(tables_router)
app.include_router(es_test.router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(docs_analytics_router)


@app.get("/")
def root():
    return {"status": "running", "service": "Agentic AI"}
