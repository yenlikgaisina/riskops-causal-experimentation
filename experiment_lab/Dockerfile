FROM python:3.12-slim
WORKDIR /app
COPY experiment_lab/requirements.txt /app/experiment_lab/requirements.txt
RUN pip install --no-cache-dir -r /app/experiment_lab/requirements.txt
COPY experiment_lab/ /app/experiment_lab/
COPY data/synthetic_user_reports.csv /app/data/synthetic_user_reports.csv
EXPOSE 8000
USER 65532:65532
CMD ["uvicorn", "experiment_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
