FROM python:3.11-slim
WORKDIR /app
COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt
COPY service /app/service
# финальный чекпойнт ViT кладётся в service/artifacts/model.pth (из MLflow)
EXPOSE 8000
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
