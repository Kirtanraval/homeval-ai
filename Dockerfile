FROM python:3.10.14-slim

WORKDIR /app

# Create non-root user for security (least privilege principle)
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Set ownership of app directory to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
