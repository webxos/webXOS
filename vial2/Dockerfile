FROM python:3.9-slim

WORKDIR /app

COPY ./vial2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./vial2/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# xAI Artifact Tags: #vial2 #mcp #docker #container #neon_mcp
