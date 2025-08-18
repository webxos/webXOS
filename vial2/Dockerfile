FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV NEON_DB_URL=postgresql://user:pass@neon.webxos.netlify.app:5432/vial2
ENV GITHUB_CLIENT_ID=your_client_id
ENV GITHUB_CLIENT_SECRET=your_client_secret

EXPOSE 8000

CMD ["uvicorn", "vial2.main:app", "--host", "0.0.0.0", "--port", "8000"]

# xAI Artifact Tags: #vial2 #docker #sqlite #neon_mcp
