FROM python:3.13-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# ---------- backend: FastAPI + PyTorch (CPU version, much smaller) ----------
FROM base AS backend
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- frontend: Streamlit only, no PyTorch ----------
FROM base AS frontend
COPY requirements-frontend.txt .
RUN pip install -r requirements-frontend.txt
COPY app/ app/
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]