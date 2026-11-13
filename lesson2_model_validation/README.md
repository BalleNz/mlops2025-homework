# MLOps 2025, lesson 2: проверка модели и метрики

Домашнее задание по курсу «Сценарии сборки моделей машинного обучения».
Студент: Зиядинов Арсен, МИВТ-25-3.

Задание: [step5_homework_model_validation_and_metrics](https://github.com/tam2511/mlops2025/blob/main/lesson2/seminar/step5_homework_model_validation_and_metrics/README.md).
База: `lesson2/seminar/step4_ge_validation_data` (пайплайн DVC + Great Expectations на датасете tips).

## Что сделано

| Задача | Файлы |
|---|---|
| 1. Poetry вместо `requirements.txt`, pre-commit: black, ruff, isort, end-of-file-fixer, check-yaml | `pyproject.toml`, `poetry.lock`, [`../.pre-commit-config.yaml`](../.pre-commit-config.yaml) |
| 2. `evaluate.py` пишет `metrics/metrics.json`: accuracy, число строк в датасете и в тестовой выборке | `src/evaluate.py` |
| 3. `validate_model.py` читает модель и данные, пересчитывает accuracy на той же тестовой выборке, сверяет её с `metrics.json` и сравнивает с `accuracy_min` из `params.yaml`. При `accuracy < accuracy_min` завершается с кодом 1 | `src/validate_model.py`, `params.yaml` |
| 4. `dvc.yaml`: у стадии `evaluate` секция `metrics`, новая стадия `validate_model` после `evaluate` | `dvc.yaml`, `dvc.lock` |

## Запуск

```bash
cd lesson2_model_validation
poetry install
poetry run pre-commit install
poetry run dvc repro
poetry run dvc metrics show
```

То же через `make install`, `make repro`, `make metrics`, проверка стиля: `make lint`.

Конфиг pre-commit лежит в корне репозитория и проверяет код всех ДЗ. DVC-проект находится в этой папке (`.dvc/` рядом с `dvc.yaml`), поэтому команды `dvc` запускаются отсюда.

## Результат

```
$ dvc metrics show
Path                  accuracy    n_rows    n_test_rows
metrics/metrics.json  0.83673     244       49
```

```
Running stage 'validate_model':
> python src/validate_model.py
Accuracy: 0.8367, accuracy_min: 0.8000
Model validation passed
```

Если поднять порог до `accuracy_min: 0.9`, пайплайн останавливается на проверке модели:

```
Running stage 'validate_model':
> python src/validate_model.py
Accuracy: 0.8367, accuracy_min: 0.9000
Model validation FAILED: accuracy is below accuracy_min
ERROR: failed to reproduce 'validate_model': failed to run: python src/validate_model.py, exited with 1
```

```
$ pre-commit run --all-files
fix end of files.........................................................Passed
check yaml...............................................................Passed
isort....................................................................Passed
black....................................................................Passed
ruff.....................................................................Passed
```

Замечание: accuracy 0.8367 равна доле класса `high_tip = False` в тестовой выборке. Логистическая регрессия из семинара на признаках `total_bill` и `size` предсказывает только `False`, поэтому порог 0.8 по сути проверяет, что модель не хуже константы. Для настоящего контроля качества к accuracy стоит добавить F1 или ROC AUC.
