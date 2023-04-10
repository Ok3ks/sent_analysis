FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["./models", "./deploy/api.py", "./"]

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]