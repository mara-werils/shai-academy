# shai.academy - Generative Image Platform

Контейнеризованная платформа для генерации изображений с FastAPI, React и PostgreSQL.

## Архитектура

- **postgres** — база данных (порт 5432)
- **fastapi_api** — REST API (порт 8000)
- **frontend_app** — React SPA через Nginx (порт 3000)

## Быстрый старт

1. Создайте файл `.env` в корне проекта и добавьте URL вашего ComfyUI на RunPod и JWT секрет:
```bash
RUNPOD_COMFY_URL=https://your-pod-id.runpod.net
JWT_SECRET=change-me
JWT_EXPIRE_MINUTES=60
```

2. Запустите все сервисы:
```bash
docker-compose up --build
```

2. Откройте в браузере:
- **Фронтенд**: http://localhost или http://localhost:3000 (оба варианта работают)
- **API Swagger**: http://localhost:8000/docs

## Использование

1. На странице Auth введите любые креды (мок-авторизация)
2. Перейдите на вкладку Generation
3. Введите промпт и нажмите Generate
4. Через 5 секунд появится сгенерированное изображение (mock)

## База данных

Все задачи сохраняются в PostgreSQL. Для просмотра через DBeaver:

- **Host**: localhost
- **Port**: 5432
- **Database**: shai_db
- **User**: shai
- **Password**: shai

Таблица `tasks` создаётся автоматически при первом запуске.

## Настройка переменных окружения

Для работы генерации изображений необходимо настроить переменную окружения `RUNPOD_COMFY_URL`:

1. **Для Docker Compose**: Создайте файл `.env` в корне проекта:
   ```
   RUNPOD_COMFY_URL=https://your-pod-id.runpod.net
   JWT_SECRET=change-me
   JWT_EXPIRE_MINUTES=60
   ```

2. **Для локальной разработки**: Создайте файл `.env` в папке `fastapi_api/`:
   ```
   RUNPOD_COMFY_URL=https://your-pod-id.runpod.net
   JWT_SECRET=change-me
   JWT_EXPIRE_MINUTES=60
   ```

## API Endpoints

- `POST /api/auth/register` — регистрация (email, name, password)
- `POST /api/auth/login` — авторизация (email, password) -> JWT access_token
- `POST /api/generate/image` — генерация изображения (требует `prompt` и `steps`, JWT)
- `POST /api/generate/video` — генерация видео (требует `prompt` и `steps`, JWT)
- `GET /api/tasks` — список последних задач


