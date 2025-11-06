# ===== Builder stage: create a small virtual environment with only the needed dependencies =====
FROM python:3.11-slim AS builder

# Build tools only in builder stage (never ship them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    # python3-pip \
    python3-venv \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv and put it on the PATH (must be before venv creation)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh 
ENV PATH="/root/.local/bin:${PATH}"


# Created a dedicated virtualenv for the final image with uv
RUN uv venv /opt/venv   
# Sets the VIRTUAL_ENV environment variable to point to the active virtual environment directory located at /opt/venv
ENV VIRTUAL_ENV=/opt/venv                   
# Updates the PATH environment variable to include the bin directory of the virtual environment, ensuring that executables installed in the virtual environment 
#are prioritized when running commands.
ENV PATH="/opt/venv/bin:${PATH}"  
# Disables pip's cache to reduce image size
ENV PIP_NO_CACHE_DIR=1                      

WORKDIR /app

# Copy only lock & metadata first for better caching
COPY pyproject.toml uv.lock ./

# Install *only dependencies* into /opt/venv (not the project yet)
# --frozen: respect uv.lock exactly
# --no-dev: omit dev deps in the container
# --no-install-project: install deps only (faster layer caching)
RUN uv sync --frozen --no-dev --no-install-project --python /opt/venv/bin/python


# ===== Runtime stage: copy the virtual environment from the builder stage and add the project files =====
FROM python:3.11-slim 

# Runtime libs only (xgboost needs libgomp), curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Bring in the ready-to-use virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Prevent Python from buffering stdout/stderr (good for logging)
ENV PYTHONUNBUFFERED=1  

# Prevent Python from writing .pyc files (not needed in containers)
ENV PYTHONDONTWRITEBYTECODE=1           
ENV PYTHONPATH=/app/src

WORKDIR /app    

# Copy the application code last (best cache behaviour)
COPY . .

# Ensure pip exists inside the copied venv, then install the project itself
# (deps already present; I avoid re-resolving with --no-deps)
RUN /opt/venv/bin/python -m ensurepip --upgrade && \
    /opt/venv/bin/python -m pip install --no-cache-dir uvicorn && \
    /opt/venv/bin/python -m pip install --no-deps -e .

# Non-root user (better security practice)
RUN useradd -u 10001 -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container healthcheck hitting the API
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

#Start the FastAPI (for dev we can use --reload outside the container)
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","2"]   
