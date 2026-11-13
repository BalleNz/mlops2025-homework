SERVICE_DIR ?= ../step2_fastapi_inference

install:
	poetry install

monitor:
	poetry run python main.py

monitor-once:
	poetry run python main.py --checks 1

monitor-with-service:
	poetry run python main.py --start-service --service-dir $(SERVICE_DIR)

test:
	poetry run pytest -q

lint:
	poetry run pre-commit run --all-files

clean:
	rm -rf logs
