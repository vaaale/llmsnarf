FROM python:3.12-slim AS backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    media-types \
    htop \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the entire project
COPY . .

# Sync dependencies using uv
RUN uv sync --frozen

# Create Documents directory
RUN mkdir -p /app/logs
RUN mkdir -p /app/traces

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["uv", "run", "llmproxy"]
