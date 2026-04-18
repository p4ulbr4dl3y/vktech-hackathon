# 🛠 Внутренний процесс разработки и тестирования (Workflow)

Этот документ описывает технические шаги, которые мы выполняем локально перед каждой отправкой (push) решения в реестр VK Tech.

---

## 1. Сборка "автономных" образов
По условиям ТЗ, интернета в контейнерах нет. Мы используем метод **Pre-baking**, чтобы включить ML-модели прямо в Docker-слои.

### Команды сборки:
Мы используем флаг `--network=host`, чтобы обойти проблемы с DNS внутри Docker-моста при скачивании моделей с Hugging Face:

```bash
# Сборка Index Service с моделью BM25
cd index && docker build --network=host -t it-academy-hackathon-solution-example-index:latest .

# Сборка Search Service
cd search && docker build --network=host -t it-academy-hackathon-solution-example-search:latest .
```

---

## 2. Локальное тестирование (Golden Set)
Для оценки качества мы используем скрипт `full_evaluator_v3.py`, который имитирует поведение проверяющей системы VK.

### Методология:
1.  **Датасет**: Используется файл `data/Go Nova.json`.
2.  **Golden Set**: Мы разметили 12 эталонных запросов (от простых текстовых до сложных технических ошибок вроде `SIGABRT`).
3.  **Метрики**:
    *   **Recall@50**: Попадает ли правильный ID в топ-50.
    *   **MRR**: На какой позиции находится первый верный ответ.
    *   **VK Score**: Взвешенная сумма (Recall * 0.8 + NDCG * 0.2).

### Запуск теста:
```bash
# Поднимаем окружение
docker compose up -d

# Запускаем бенчмарк
export OPEN_API_LOGIN=... 
export OPEN_API_PASSWORD=...
python3 full_evaluator_v3.py
```

---

## 3. Решение технических проблем

### Лимиты контекста (8192 токена)
Чтобы избежать ошибки "Context limit exceeded", мы внедрили **Truncation**:
*   В `index/main.py`: любой контент чанка обрезается до **5,000 символов** перед отправкой в Dense-модель.
*   В `search/main.py`: поисковый запрос (включая HyDE-расширение) также обрезается до **5,000 символов**.

### Ошибка 429 (Rate Limits)
Для защиты от блокировки со стороны API VK:
*   Снижен `RERANK_LIMIT` до **35-50**.
*   Внедрен цикл **Retries** с экспоненциальной задержкой (3-6-9 секунд) при получении HTTP 429.

---

## 4. Процесс сдачи (Push)
Когда локальный бенчмарк показывает SCORE > 0.9, мы выполняем пуш в реестр:

```bash
docker login 83.166.249.64:5000 -u <login> -p <password>

# Пушим тегированные образы
docker tag it-academy-hackathon-solution-example-index:latest 83.166.249.64:5000/29502/index-service:latest
docker push 83.166.249.64:5000/29502/index-service:latest

docker tag it-academy-hackathon-solution-example-search:latest 83.166.249.64:5000/29502/search-service:latest
docker push 83.166.249.64:5000/29502/search-service:latest
```

---
**Текущий статус:** Локально подтвержден Score 0.93 на расширенном наборе тестов.
