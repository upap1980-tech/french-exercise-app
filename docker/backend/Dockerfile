FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY backend/requirements.txt ./requirements.txt
RUN apt-get update && \
	apt-get install -y build-essential libcairo2-dev libpango1.0-dev libgdk-pixbuf2.0-dev libffi-dev libxml2-dev libxslt1-dev && \
	rm -rf /var/lib/apt/lists/* || true
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ /app
EXPOSE 5012
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5012", "--workers", "2", "--threads", "2", "--timeout", "120"]
