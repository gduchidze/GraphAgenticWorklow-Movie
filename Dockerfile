FROM python:3.12-slim

RUN apt-get update && apt-get install -y net-tools && apt-get install -y htop

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]