#!/bin/bash

# Остановить существующий контейнер, если есть
docker stop cypher-app || true
docker rm cypher-app || true

# Собрать образ
echo "Building Docker image..."
docker build -t cypher-app -f Dockerfile.local .

# Запустить контейнер
echo "Starting container..."
docker run -d \
    --name cypher-app \
    -p 8000:8000 \
    --restart unless-stopped \
    cypher-app

echo "App is running on http://localhost:8000"
echo "To view logs: docker logs -f cypher-app"
echo "To stop: docker stop cypher-app" 