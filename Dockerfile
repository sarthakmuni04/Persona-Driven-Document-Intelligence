FROM python:3.9-slim
WORKDIR /app
COPY app/ .
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir sentence-transformers faiss-cpu transformers PyMuPDF==1.22.3
CMD ["python", "main.py"]
