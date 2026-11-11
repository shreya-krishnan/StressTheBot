from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.rules import router as rules_router
from app.api.routes.experiments import router as experiments_router
from app.database.connection import init_db


app = FastAPI(
    title="StressTheBot API",
    description="Clinical AI Safety Stress-Testing Platform",
    version="1.0.0"
)


@app.on_event("startup")
def initialize_database():
    init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(rules_router)
app.include_router(experiments_router)


@app.get("/")
def home():

    return {
        "message": "StressTheBot API is running",
        "status": "active"
    }


@app.get("/health")
def health():

    return {"status": "healthy"}
