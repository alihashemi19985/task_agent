FROM docker.arvancloud.ir/python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies with Docker layer caching.
COPY requirement.txt /tmp/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --timeout 120 \
    --retries 5 \
    -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Non-root runtime user.
ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" app \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home app

COPY --chown=app:app . /app

RUN mkdir -p /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
