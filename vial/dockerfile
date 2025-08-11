FROM python:3.11-slim

WORKDIR /app

COPY vial/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vial/ .
COPY errorlog.md /app/errorlog.md

EXPOSE 5000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5000"]
