FROM apache/airflow:2.9.2-python3.12

USER root
RUN sudo apt-get update
RUN sudo apt-get install -y libgomp1 git

USER airflow
COPY --chown=airflow:root pyproject.toml /opt/ml-churn/pyproject.toml
COPY --chown=airflow:root src/ /opt/ml-churn/src/

RUN pip install --no-cache-dir "/opt/ml-churn[train]" "dvc[s3]"
RUN git config --global safe.directory /opt/airflow/project
