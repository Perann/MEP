# Utilisation d'une base Python stable (Debian Bookworm)
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash

ENV PATH="/root/.pixi/bin:$PATH"


WORKDIR /app


COPY pixi.toml pixi.lock ./


RUN pixi install --frozen

COPY . .

EXPOSE 8000


CMD ["pixi", "run", "dev"]