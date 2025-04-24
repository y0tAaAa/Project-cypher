FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV FLASK_APP=server.py
ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "server:app"]