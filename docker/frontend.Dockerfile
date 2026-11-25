FROM python:3.12-slim
WORKDIR /app
COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY frontend .
RUN useradd -m app
USER app
ENV BACKEND_URL=http://backend:5000
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
