FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "db/wait-for-it.sh", "mongo:27017", "--", "python", "vial/unified_server.py"]
