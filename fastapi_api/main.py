import asyncio
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://shai:shai@postgres:5432/shai_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    prompt = Column(Text, nullable=False)
    user_id = Column(String(64), nullable=True)
    type = Column(String(16), nullable=False)  # image | video
    status = Column(String(32), nullable=False, default="completed")
    result_url = Column(Text, nullable=False)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="FastAPI Generative API", version="0.5.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    user_id: Optional[str] = Field(
        None, description="Optional user identifier for tracking"
    )


class GenerateResponse(BaseModel):
    task_id: str
    status: str
    image_url: str


class GenerateVideoResponse(BaseModel):
    task_id: str
    status: str
    message: str
    video_url: str


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.get("/api/tasks")
async def list_tasks():
    try:
        with SessionLocal() as db:
            rows = (
                db.query(Task)
                .order_by(Task.id.desc())
                .limit(50)
                .all()
            )
            return [
                {
                    "task_id": r.task_id,
                    "prompt": r.prompt,
                    "user_id": r.user_id,
                    "type": r.type,
                    "status": r.status,
                    "result_url": r.result_url,
                }
                for r in rows
            ]
    except SQLAlchemyError as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="DB error while fetching tasks") from exc


@app.post("/api/generate/image", response_model=GenerateResponse)
async def generate_image(payload: GenerateRequest):
    await asyncio.sleep(5)
    task_id = f"mock_task_{int(time.time())}"
    image_url = (
        "https://placehold.co/1024x1024/004c99/ffffff/png?text=Generated+Image+MOCK"
    )
    try:
        with SessionLocal() as db:
            db_task = Task(
                task_id=task_id,
                prompt=payload.prompt,
                user_id=payload.user_id,
                type="image",
                status="completed",
                result_url=image_url,
            )
            db.add(db_task)
            db.commit()
    except SQLAlchemyError as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="DB error while saving task") from exc

    return GenerateResponse(task_id=task_id, status="completed", image_url=image_url)


@app.post("/api/generate/video", response_model=GenerateVideoResponse)
async def generate_video(payload: GenerateRequest):
    await asyncio.sleep(10)
    task_id = f"mock_video_{int(time.time())}"
    video_url = "https://placehold.co/600x400/cc0000/ffffff/png?text=Video+MOCK"
    try:
        with SessionLocal() as db:
            db_task = Task(
                task_id=task_id,
                prompt=payload.prompt,
                user_id=payload.user_id,
                type="video",
                status="completed",
                result_url=video_url,
            )
            db.add(db_task)
            db.commit()
    except SQLAlchemyError as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="DB error while saving task") from exc

    return GenerateVideoResponse(
        task_id=task_id,
        status="completed",
        message="Video generation mock completed.",
        video_url=video_url,
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(auth: AuthRequest):
    return AuthResponse(access_token="mock_jwt_token")

