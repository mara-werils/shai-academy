import asyncio
import os
import time
import copy
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import requests
from jose import jwt, JWTError
from passlib.context import CryptContext

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Получаем URL ComfyUI из переменной окружения
COMFY_URL = os.getenv("RUNPOD_COMFY_URL")
SECRET_KEY = os.getenv("JWT_SECRET", "change-me-please")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="shai.academy API", version="1.0.0")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg2://shai:shai@postgres:5432/shai_db"
)

# Создаем engine с пулом соединений и настройками для retry
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Проверяет соединение перед использованием
    pool_recycle=300,    # Переиспользует соединения
    connect_args={"connect_timeout": 10}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    prompt = Column(Text)
    user_id = Column(String, nullable=True)
    type = Column(String)  # 'image' or 'video'
    status = Column(String)
    result_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# Функция для инициализации БД с retry логикой
def init_db():
    """Инициализирует БД с повторными попытками подключения"""
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database initialized successfully")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {str(e)}")
                logger.warning("Application will continue, but database operations may fail")


# Инициализируем БД при старте (с retry)
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


FORBIDDEN_TERMS = {
    # Sexual explicit
    "sexual",
    "sex",
    "porn",
    "pornographic",
    "pornography",
    "nude",
    "naked",
    "nsfw",
    "explicit",
    "fetish",
    "genitals",
    # Violence / self-harm
    "violence",
    "violent",
    "gore",
    "blood",
    "beheading",
    "suicide",
    "self-harm",
    "murder",
    "kill",
    "killing",
    # Hate / harassment
    "hate",
    "racist",
    "racism",
    "homophobic",
    "slur",
    "abuse",
    "harass",
    "terror",
    "terrorist",
    # Child exploitation (hard block)
    "child porn",
    "child sexual",
    "csam",
    "loli",
    "underage",
}


def check_prompt_guardrails(prompt: str):
    text = prompt.lower()
    for term in FORBIDDEN_TERMS:
        if term in text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Prompt rejected by safety filter (contains '{term}').",
            )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(db, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_current_user(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception
    return user


def get_comfyui_output_url(prompt_id: str, max_wait: int = 120, allowed_ext: Optional[list[str]] = None) -> Optional[str]:
    """
    Опрашивает ComfyUI API для получения URL сгенерированного файла (изображение/видео).
    """
    if allowed_ext is None:
        allowed_ext = ["png", "jpg", "jpeg", "webp"]

    start_time = time.time()
    poll_interval = 2  # Опрашиваем каждые 2 секунды
    
    logger.info(f"Starting to poll for prompt_id: {prompt_id}")
    
    while time.time() - start_time < max_wait:
        try:
            # Проверяем историю выполнения
            history_url = f"{COMFY_URL.rstrip('/')}/history/{prompt_id}"
            response = requests.get(history_url, timeout=10)
            
            if response.status_code == 200:
                history_data = response.json()
                logger.debug(f"History response: {json.dumps(history_data, indent=2)}")
                
                # ComfyUI может возвращать историю в разных форматах
                # Формат 1: {prompt_id: {status: [...], outputs: {...}}}
                # Формат 2: {prompt_id: [{status: {...}, outputs: {...}}]}
                
                task_data = None
                if prompt_id in history_data:
                    task_data = history_data[prompt_id]
                elif isinstance(history_data, dict):
                    # Может быть вложенная структура
                    for key, value in history_data.items():
                        if key == prompt_id or (isinstance(value, dict) and prompt_id in str(value)):
                            task_data = value
                            break
                
                if task_data:
                    # Если task_data - список, берем последний элемент
                    if isinstance(task_data, list) and len(task_data) > 0:
                        task_data = task_data[-1]
                    
                    # Проверяем, есть ли выходные данные (outputs)
                    if isinstance(task_data, dict) and "outputs" in task_data:
                        outputs = task_data["outputs"]

                        def pick_file(record: dict) -> Optional[str]:
                            filename = record.get("filename", "")
                            subfolder = record.get("subfolder", "")
                            file_type = record.get("type", "output")
                            if not filename:
                                return None
                            ext = filename.split(".")[-1].lower()
                            if ext not in allowed_ext:
                                return None
                            view_url = f"{COMFY_URL.rstrip('/')}/view"
                            params = {"filename": filename, "type": file_type}
                            if subfolder:
                                params["subfolder"] = subfolder
                            return f"{view_url}?{urlencode(params)}"

                        # Ищем в images/files/video
                        for node_id, node_output in outputs.items():
                            if isinstance(node_output, dict):
                                for key in ["images", "files", "video"]:
                                    if key in node_output:
                                        items = node_output[key]
                                        if items and len(items) > 0:
                                            url = pick_file(items[0])
                                            if url:
                                                logger.info(f"Found output URL: {url}")
                                                return url
                    
                    # Если outputs нет, но есть status, проверяем статус
                    if isinstance(task_data, dict) and "status" in task_data:
                        status = task_data["status"]
                        if isinstance(status, list) and len(status) > 0:
                            # Берем последний статус
                            last_status = status[-1]
                            if isinstance(last_status, dict) and last_status.get("completed", False):
                                # Задача завершена, но outputs может быть в другом месте
                                logger.debug("Task completed, but outputs not found yet")
            
            # Если задача еще не завершена, ждем и пробуем снова
            logger.debug(f"Task {prompt_id} still processing, waiting {poll_interval}s...")
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error polling ComfyUI: {str(e)}, retrying...")
            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Unexpected error while polling: {str(e)}", exc_info=True)
            time.sleep(poll_interval)
    
    logger.warning(f"Timeout waiting for prompt_id {prompt_id} after {max_wait}s")
    return None


class UserBase(BaseModel):
    email: str = Field(..., description="User email")
    name: str = Field(..., min_length=1, description="User name")


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=72, description="User password")


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    steps: int = Field(20, ge=1, le=100, description="Number of sampling steps")
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


class TaskResponse(BaseModel):
    id: int
    task_id: str
    prompt: str
    user_id: Optional[str]
    type: str
    status: str
    result_url: str
    created_at: datetime


# Шаблон ComfyUI workflow (обновленный по новому JSON)
COMFY_WORKFLOW_TEMPLATE = {
    "3": {
        "inputs": {
            "seed": 891982008105110,  # будет переопределен на случайный
            "steps": 30,  # заменяется на steps пользователя
            "cfg": 7,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "denoise": 1,
            "model": ["10", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    },
    "4": {
        "inputs": {
            "ckpt_name": "dreamshaper_8.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "text": "masterpiece, best quality, ultra-detailed, 8K, RAW photo, intricate details, stunning visuals,upper-body, highly detailed face, smooth skin, white colthes, realistic lighting, beautiful Chinese girl, solo, traditional Chinese dress, golden embroidery, elegant, black hair, delicate hair ornament, cinematic lighting, soft focus,(white background:1.05)",  # заменяется на prompt пользователя
            "clip": ["10", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "(low quality, worst quality:1.4), (blurry:1.2), (bad anatomy:1.3), extra limbs, deformed, watermark, text, signature, bareness",
            "clip": ["10", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        },
        "class_type": "VAEDecode"
    },
    "9": {
        "inputs": {
            "filename_prefix": "2loras_test_",
            "images": ["8", 0]
        },
        "class_type": "SaveImage"
    },
    "10": {
        "inputs": {
            "model": ["11", 0],
            "clip": ["11", 1],
            "lora_name": "add-detail-xl.safetensors",
            "strength_model": 1,
            "strength_clip": 1
        },
        "class_type": "LoraLoader",
        "properties": {
            "models": [
                {
                    "name": "MoXinV1.safetensors",
                    "url": "https://civitai.com/api/download/models/14856?type=Model&format=SafeTensor&size=full&fp=fp16",
                    "directory": "loras"
                }
            ]
        },
        "widgets_values": [
            "add-detail-xl.safetensors",
            1,
            1
        ]
    },
    "11": {
        "inputs": {
            "model": ["4", 0],
            "clip": ["4", 1],
            "lora_name": "blindbox_v1_mix.safetensors",
            "strength_model": 0.9,
            "strength_clip": 1
        },
        "class_type": "LoraLoader",
        "properties": {
            "models": [
                {
                    "name": "blindbox_v1_mix.safetensors",
                    "url": "https://civitai.com/api/download/models/32988?type=Model&format=SafeTensor&size=full&fp=fp16",
                    "directory": "loras"
                }
            ]
        },
        "widgets_values": [
            "blindbox_v1_mix.safetensors",
            0.9,
            1
        ]
    }
}

# Шаблон ComfyUI workflow для видео (по предоставленному JSON)
COMFY_VIDEO_WORKFLOW_TEMPLATE = {
    "3": {
        "inputs": {
            "seed": 575373598559925,
            "steps": 30,  # заменяется на steps пользователя
            "cfg": 6,
            "sampler_name": "uni_pc",
            "scheduler": "simple",
            "denoise": 1,
            "model": ["48", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["40", 0]
        },
        "class_type": "KSampler"
    },
    "6": {
        "inputs": {
            "text": "a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera",
            "clip": ["38", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            "clip": ["38", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["39", 0]
        },
        "class_type": "VAEDecode"
    },
    "28": {
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "ComfyUI_video",
            "lossless": False,
            "quality": 90,
            "fps": 16,
            "method": "default"
        },
        "class_type": "SaveAnimatedWEBP",
        "widgets_values": [
            "ComfyUI",
            16,
            False,
            90,
            "default"
        ]
    },
    "39": {
        "inputs": {
            "vae_name": "wan_2.1_vae.safetensors"
        },
        "class_type": "VAELoader",
        "widgets_values": ["wan_2.1_vae.safetensors"]
    },
    "40": {
        "inputs": {
            "width": 832,
            "height": 480,
            "length": 33,
            "batch_size": 1
        },
        "class_type": "EmptyHunyuanLatentVideo",
        "widgets_values": [
            832,
            480,
            33,
            1
        ]
    },
    "47": {
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "ComfyUI_video",
            "codec": "vp9",
            "fps": 24,
            "crf": 32
        },
        "class_type": "SaveWEBM",
        "widgets_values": [
            "ComfyUI",
            "vp9",
            24,
            32
        ]
    },
    "48": {
        "inputs": {
            "model": ["37", 0],
            "shift": 8
        },
        "class_type": "ModelSamplingSD3",
        "widgets_values": [8]
    },
    "37": {
        "inputs": {
            "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
            "weight_dtype": "default"
        },
        "class_type": "UNETLoader",
        "widgets_values": [
            "wan2.1_t2v_1.3B_fp16.safetensors",
            "default"
        ]
    },
    "38": {
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan"
        },
        "class_type": "CLIPLoader",
        "widgets_values": [
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "wan",
            "default"
        ]
    }
}


@app.post("/api/generate/image", response_model=GenerateResponse)
async def generate_image(payload: GenerateRequest, current_user: User = Depends(get_current_user)):
    check_prompt_guardrails(payload.prompt)
    if not COMFY_URL:
        raise HTTPException(
            status_code=500,
            detail="COMFY_URL not configured. Please set RUNPOD_COMFY_URL environment variable."
        )
    
    workflow = copy.deepcopy(COMFY_WORKFLOW_TEMPLATE)
    workflow["3"]["inputs"]["steps"] = payload.steps
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    workflow["6"]["inputs"]["text"] = payload.prompt
    
    client_id = f"shai_academy_{int(time.time())}"
    prompt_data = {
        "prompt": workflow,
        "client_id": client_id
    }
    
    logger.info(f"Sending request to ComfyUI: {COMFY_URL}")
    logger.info(f"Client ID: {client_id}")
    logger.debug(f"Workflow data: {json.dumps(workflow, indent=2)}")
    
    try:
        comfy_url = f"{COMFY_URL.rstrip('/')}/prompt"
        logger.info(f"Trying URL: {comfy_url}")
        response = requests.post(
            comfy_url,
            json=prompt_data,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        response_text = response.text
        logger.debug(f"Response text: {response_text}")
        if response.status_code != 200:
            error_detail = f"ComfyUI returned status {response.status_code}"
            try:
                error_json = response.json()
                logger.error(f"ComfyUI error JSON: {json.dumps(error_json, indent=2)}")
                if isinstance(error_json, dict):
                    if "error" in error_json:
                        error_detail += f": {error_json['error']}"
                    elif "message" in error_json:
                        error_detail += f": {error_json['message']}"
                    else:
                        error_detail += f": {error_json}"
            except Exception:
                error_detail += f": {response_text[:500]}"
            raise HTTPException(status_code=502, detail=error_detail)
        
        try:
            comfy_response = response.json()
            logger.info(f"ComfyUI response: {json.dumps(comfy_response, indent=2)}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response: {response.text}")
            raise HTTPException(
                status_code=502,
                detail=f"ComfyUI returned invalid JSON: {response.text[:500]}"
            )
        
        task_id = comfy_response.get("prompt_id", f"comfy_task_{int(time.time())}")
        logger.info(f"ComfyUI accepted prompt with ID: {task_id}")
        
        logger.info("Polling ComfyUI for result...")
        image_url = get_comfyui_output_url(task_id, max_wait=120)
        
        db = SessionLocal()
        try:
            if image_url:
                status = "completed"
                result_url = image_url
                logger.info(f"Image generated successfully: {image_url}")
            else:
                status = "processing"
                result_url = ""
                logger.warning(f"Image not ready yet for task_id: {task_id}")
            
            task = Task(
                task_id=task_id,
                prompt=payload.prompt,
                user_id=str(current_user.id),
                type="image",
                status=status,
                result_url=result_url,
            )
            db.add(task)
            db.commit()
        finally:
            db.close()
        
        if image_url:
            return GenerateResponse(
                task_id=task_id,
                status="completed",
                image_url=image_url
            )
        else:
            placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
            return GenerateResponse(
                task_id=task_id,
                status="processing",
                image_url=placeholder_url
            )
        
    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to ComfyUI")
        raise HTTPException(
            status_code=502,
            detail="ComfyUI request timed out. Please check if the service is running."
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to ComfyUI at {COMFY_URL}. Please check the URL and ensure the service is running."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        error_detail = f"ComfyUI returned status {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_detail += f": {error_json['error']}"
            elif "message" in error_json:
                error_detail += f": {error_json['message']}"
        except:
            error_detail += f": {e.response.text[:500]}"
        raise HTTPException(
            status_code=502,
            detail=error_detail
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to ComfyUI: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/generate/video", response_model=GenerateVideoResponse)
async def generate_video(payload: GenerateRequest, current_user: User = Depends(get_current_user)):
    check_prompt_guardrails(payload.prompt)
    if not COMFY_URL:
        raise HTTPException(
            status_code=500,
            detail="COMFY_URL not configured. Please set RUNPOD_COMFY_URL environment variable."
        )

    workflow = copy.deepcopy(COMFY_VIDEO_WORKFLOW_TEMPLATE)
    workflow["3"]["inputs"]["steps"] = payload.steps
    workflow["3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    workflow["6"]["inputs"]["text"] = payload.prompt

    client_id = f"shai_academy_vid_{int(time.time())}"
    prompt_data = {
        "prompt": workflow,
        "client_id": client_id
    }

    logger.info(f"Sending video request to ComfyUI: {COMFY_URL}")
    logger.info(f"Client ID: {client_id}")
    logger.debug(f"Video workflow data: {json.dumps(workflow, indent=2)}")

    try:
        comfy_url = f"{COMFY_URL.rstrip('/')}/prompt"
        logger.info(f"Trying URL: {comfy_url}")
        response = requests.post(
            comfy_url,
            json=prompt_data,
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        response_text = response.text
        logger.debug(f"Response text: {response_text}")
        if response.status_code != 200:
            error_detail = f"ComfyUI returned status {response.status_code}"
            try:
                error_json = response.json()
                logger.error(f"ComfyUI error JSON: {json.dumps(error_json, indent=2)}")
                if isinstance(error_json, dict):
                    if "error" in error_json:
                        error_detail += f": {error_json['error']}"
                    elif "message" in error_json:
                        error_detail += f": {error_json['message']}"
                    else:
                        error_detail += f": {error_json}"
            except Exception:
                error_detail += f": {response_text[:500]}"
            raise HTTPException(status_code=502, detail=error_detail)

        comfy_response = response.json()
        logger.info(f"ComfyUI response: {json.dumps(comfy_response, indent=2)}")
        task_id = comfy_response.get("prompt_id", f"comfy_video_{int(time.time())}")

        # Ждем готовности видео (webm/mp4/webp/png)
        video_url = get_comfyui_output_url(
            task_id,
            max_wait=180,
            allowed_ext=["webm", "mp4", "webp", "png"]
        )

        db = SessionLocal()
        try:
            status = "completed" if video_url else "processing"
            result_url = video_url or ""
            task = Task(
                task_id=task_id,
                prompt=payload.prompt,
                user_id=str(current_user.id),
                type="video",
                status=status,
                result_url=result_url,
            )
            db.add(task)
            db.commit()
        finally:
            db.close()

        if video_url:
            return GenerateVideoResponse(
                task_id=task_id,
                status="completed",
                message="Video generated successfully",
                video_url=video_url,
            )
        else:
            placeholder_url = f"https://placehold.co/512x512/ff7849/ffffff/png?text=Video+Processing+{task_id[:8]}"
            return GenerateVideoResponse(
                task_id=task_id,
                status="processing",
                message="Video is still processing",
                video_url=placeholder_url,
            )

    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to ComfyUI (video)")
        raise HTTPException(
            status_code=502,
            detail="ComfyUI video request timed out. Please check if the service is running."
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error (video): {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to ComfyUI at {COMFY_URL}. Please check the URL and ensure the service is running."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error (video): {e.response.status_code} - {e.response.text}")
        error_detail = f"ComfyUI returned status {e.response.status_code}"
        try:
            error_json = e.response.json()
            if "error" in error_json:
                error_detail += f": {error_json['error']}"
            elif "message" in error_json:
                error_detail += f": {error_json['message']}"
        except Exception:
            error_detail += f": {e.response.text[:500]}"
        raise HTTPException(status_code=502, detail=error_detail)
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error (video): {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to ComfyUI: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error (video): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/tasks", response_model=list[TaskResponse])
async def get_tasks():
    db = SessionLocal()
    try:
        tasks = db.query(Task).order_by(Task.id.desc()).limit(50).all()
        return tasks
    finally:
        db.close()


@app.post("/api/auth/register", response_model=AuthResponse, status_code=201)
def register(user: UserCreate, db: SessionLocal = Depends(get_db)):
    if len(user.password) > 72:
        raise HTTPException(status_code=400, detail="Password too long (max 72 characters)")
    existing = get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = get_password_hash(user.password)
    new_user = User(email=user.email, name=user.name, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": str(new_user.id)})
    return AuthResponse(access_token=token)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(auth: UserLogin, db: SessionLocal = Depends(get_db)):
    user = get_user_by_email(db, auth.email)
    if not user or not verify_password(auth.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return AuthResponse(access_token=token)


@app.get("/api/tasks/{task_id}/status", response_model=GenerateResponse)
async def get_task_status(task_id: str):
    """
    Опрашивает статус задачи и возвращает URL изображения, если оно готово.
    """
    if not COMFY_URL:
        raise HTTPException(
            status_code=500,
            detail="COMFY_URL not configured. Please set RUNPOD_COMFY_URL environment variable."
        )
    
    # Проверяем БД
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.task_id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Если задача уже завершена и есть URL, возвращаем его
        if task.status == "completed" and task.result_url:
            return GenerateResponse(
                task_id=task.task_id,
                status="completed",
                image_url=task.result_url
            )
        
        # Если задача еще обрабатывается, опрашиваем ComfyUI
        if task.status == "processing":
            image_url = get_comfyui_image_url(task_id, max_wait=5)  # Короткий опрос
            
            if image_url:
                # Обновляем задачу в БД
                task.status = "completed"
                task.result_url = image_url
                db.commit()
                
                return GenerateResponse(
                    task_id=task.task_id,
                    status="completed",
                    image_url=image_url
                )
            else:
                # Все еще обрабатывается
                placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
                return GenerateResponse(
                    task_id=task.task_id,
                    status="processing",
                    image_url=placeholder_url
                )
        
        # Если задача завершена, но нет URL (не должно быть, но на всякий случай)
        placeholder_url = f"https://placehold.co/512x512/3b82f6/ffffff/png?text=Processing+{task_id[:8]}"
        return GenerateResponse(
            task_id=task.task_id,
            status=task.status,
            image_url=placeholder_url
        )
        
    finally:
        db.close()


