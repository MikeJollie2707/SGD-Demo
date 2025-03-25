FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

ADD . .

CMD [ "uv", "run", "main.py" ]
