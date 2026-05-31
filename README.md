# mipt-ml-deploy

Деплой ML-системы предсказания оттока клиентов телеком-оператора
(Telco Customer Churn).

Учебный проект курса по развёртыванию ML-систем (МФТИ).

## Решение

Описание решения — [docs/00-solution.md](docs/00-solution.md).

## Структура

```
docs/              описание решения
src/ml_churn/      обучение модели
src/ml_service/    сервис инференса (API, фичи, метрики, загрузка модели)
infra/
  terraform/       создание VM (Yandex Cloud)
  docker-compose   сервисы
  dags/            Airflow DAG обучения
  monitoring/      Prometheus + Grafana (дашборды, алерты)
notebooks/         исследовательский ноутбук
data/              данные под DVC
```