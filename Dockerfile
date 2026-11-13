# Pinned base image digest for reproducible builds.
FROM python:3.11-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --index-url https://pypi.org/simple -r requirements.txt

COPY harness ./harness
COPY run_agent.py ./

# The evaluator runs:
#   python /app/run_agent.py --tasks-dir /data/tasks
# It may override the entrypoint, so the script must work both ways.
ENTRYPOINT ["python", "/app/run_agent.py"]
