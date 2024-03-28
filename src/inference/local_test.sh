curl -X 'POST' \
    'http://127.0.0.1:8000/invocations' \
    -F 'file=@data/dataset.tfrecord;type=application/octet-stream' \
    --output output.csv

curl -X POST \
    -F "file=@data/dataset.tfrecord" \
    "http://localhost:8080/invocations" \
    --output output.csv
