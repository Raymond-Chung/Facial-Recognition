FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/
ENV UV_NO_DEV=1


COPY /camera/haarcascade_frontalface_default.xml /app
COPY /camera/facial-rec.py /app
COPY /image-recognition/emotion_model.keras /app

WORKDIR /app
RUN uv sync --locked

CMD ["uv", "run", "facial-rec.py"]