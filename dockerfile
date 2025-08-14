FROM python:3.11-slim
WORKDIR /app
COPY main/server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main/server/ .
COPY main/server/db/wait-for-it.sh .
RUN chmod +x wait-for-it.sh
RUN openssl req -x509 -newkey rsa:4096 -nodes -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/C=US/ST=State/L=City/O=Org/OU=Unit/CN=localhost"
EXPOSE 8000
CMD ["./wait-for-it.sh","redis:6379","--","python","unified_server.py"]
