FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY app-requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app.py ./app.py
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser reports/figures/ ./reports/figures/
COPY --chown=appuser:appuser data/raw/Raw_Breast_Cancer_Data.csv ./data/raw/Raw_Breast_Cancer_Data.csv

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
