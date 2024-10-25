docker build -t epitome_image --platform linux/arm64 .
docker run -it epitome_image
python main.py