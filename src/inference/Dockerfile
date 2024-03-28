# Use the slim version of Python 3.10 as the base image
FROM python:3.10-slim
# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
# Install gcc to compile any extensions
RUN apt-get update && apt-get install -y gcc

# Copy the necessary files into the container at /app
COPY dataloader.py /app/dataloader.py
COPY deepseq.py /app/deepseq.py
COPY inference.py /app/inference.py
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
COPY entrypoint.sh /app/entrypoint.sh

# Set the working directory to /app
WORKDIR /app

# ONLY FOR TESTING: Copy pretrained model
COPY data/pretrained_weight.pth /opt/ml/model/pretrained_weight.pth


# Make entrypoint.sh executable
RUN chmod +x entrypoint.sh

# Install FastAPI, Uvicorn, and other required packages
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

# Expose the port FastAPI will run on
EXPOSE 8080

# Set the entry point to use the entry script
ENTRYPOINT ["./entrypoint.sh"]
