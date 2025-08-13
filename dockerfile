FROM python:3.8-slim

WORKDIR /app

COPY db/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "vial.unified_server:app", "--host", "0.0.0.0", "--port", "8000"]
