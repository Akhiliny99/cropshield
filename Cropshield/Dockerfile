
FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY cropshield.onnx .
COPY cropshield.onnx.data .
COPY class_names.json .


COPY main.py .
COPY app.py .


EXPOSE 8000
EXPOSE 8501


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
