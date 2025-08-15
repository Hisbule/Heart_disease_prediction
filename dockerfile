FROM python:3.9-slim

WORKDIR /app


COPY model/heart_model.joblib /app/model/heart_model.joblib


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./Fast_api ./Fast_api
COPY ./model ./model

CMD ["uvicorn", "Fast_api.main:app", "--host","0.0.0.0" ,"--port", "8000", "--reload"]
