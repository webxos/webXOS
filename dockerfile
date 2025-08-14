FROM python:3.10-slim

WORKDIR /app

COPY main/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main/server /app/main/server
COPY main/pages /app/main/pages
COPY certs /app/certs
COPY notes /app/notes
COPY vial_mcp.db /app/vial_mcp.db
COPY backups /app/backups

RUN mkdir -p /app/main/server/vial/static
COPY main/server/vial/static /app/main/server/vial/static

EXPOSE 8000

CMD ["uvicorn", "main.server.unified_server:app", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "/app/certs/server.key", "--ssl-certfile", "/app/certs/server.crt"]
