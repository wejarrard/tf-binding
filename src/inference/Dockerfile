# download python 3.10
FROM python:3.10-slim

# Install gcc
RUN apt-get update && apt-get install -y gcc

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "inference.py"]