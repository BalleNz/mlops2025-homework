install:
	poetry install

services:
	docker compose up -d --build

pull:
	docker exec ollama_server ollama pull qwen2.5:1.5b

pose-api:
	poetry run uvicorn services.pose_api:app --host 127.0.0.1 --port 8001

run:
	poetry run python main.py "танец макарена"

test:
	poetry run pytest -q

lint:
	poetry run pre-commit run --all-files

stop:
	docker compose down
