# shai.academy Generative Studio (FastAPI + React, zero-cost mock + Postgres)

Контейнеризованная платформа с мок-генерацией: FastAPI не делает внешних вызовов, имитирует задержку и возвращает заглушки (без биллинга). Фронтенд — React/Vite.

## Сервисы
- `fastapi_api` — REST API (порт 8000), мок-генерация изображений/видео (без внешних API), пишет задачи в Postgres.
- `frontend_app` — React/Vite SPA, отдаётся через Nginx (порт 3000), проксирует `/api` на `fastapi_api`.
- `postgres` — база данных (порт 5432), пользователи/БД по умолчанию: `shai/shai`, БД `shai_db`.

## Подготовка / Запуск
```
docker-compose up --build
```

## API
- Swagger: `http://localhost:8000/docs`
- Генерация:
```
curl -X POST http://localhost:8000/api/generate/image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "cat in space", "user_id": "demo"}'
```
Ответ: `{"task_id": "mock_task_<ts>", "status": "completed", "image_url": "https://placehold.co/1024x1024/004c99/ffffff/png?text=Generated+Image+MOCK"}`

- История (последние 50): `GET http://localhost:8000/api/tasks`

- Мок-логин:
```
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo"}'
```
Ответ: `{"access_token": "mock_jwt_token", "token_type": "bearer"}`.

## Фронтенд
- Откройте `http://localhost:3000`.
- Вкладка Auth: введите любые креды (мок), токен сохранится в localStorage.
- Вкладка Generation: введите промпт → `Generate` → через ~5 секунд появится заглушка-изображение; можно скачать. История запросов хранится в памяти сессии.

## DBeaver / подключение к БД
- Host: `localhost`
- Port: `5432`
- DB: `shai_db`
- User: `shai`
- Password: `shai`
- SSL: off (локально)

