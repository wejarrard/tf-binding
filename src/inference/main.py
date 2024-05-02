# local testings
# uvicorn main:app --reload
# curl -X 'POST' \
#   'http://127.0.0.1:8000/invocations' \
#   -F 'file=@data/dataset_2.tfrecord;type=application/octet-stream' \
#   --output downloaded_file.tsv
import io
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import uvicorn
from dataloader import EnhancedTFRecordDataset
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from inference import load_model, predict


class ExecutionParameters(BaseModel):
    MaxConcurrentTransforms: int = 1
    BatchStrategy: str = "MULTI_RECORD"
    MaxPayloadInMB: int = 100


app = FastAPI()


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


@app.get("/ping")
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    model_path = os.getenv("MODEL_PATH")
    model = load_model(model_path, num_tfs=1)
    health = model is not None  # You can insert a health check here

    status = 200 if health else 404
    return Response(content="\n", status_code=status, media_type="application/json")


# Execution-parameters endpoint
@app.get("/execution-parameters", response_model=ExecutionParameters)
async def execution_parameters():
    # You can dynamically adjust these values based on the environment or other logic
    params = ExecutionParameters()
    return params


@app.post("/invocations")
async def get_inference(file: UploadFile):
    print("Received file")
    # Load your model (consider loading it outside of request handling if it's heavy)
    # model = load_model("/opt/ml/model")
    if os.getenv("MODEL_PATH"):
        model_path = os.getenv("MODEL_PATH")
    else:
        model_path = "data/"
    model = load_model(model_path, num_tfs=1)

    data_path = save_upload_file_tmp(file)

    index_path = None
    description = {
        "input": "byte",
        "target": "byte",
        "weight": "byte",
        "chr_name": "byte",
        "start": "int",
        "end": "int",
        "cell_line": "byte",
        # "tf_list": "byte",
    }

    # CHeck if file path exists
    if not os.path.exists(data_path):
        print(f"File path {data_path} does not exist.")
        return
    else:
        print(f"File path {data_path} exists.")

    dataset = EnhancedTFRecordDataset(
        data_path=data_path,
        index_path=index_path,
        description=description,
        compression_type="gzip",
        num_tfs=1,
    )

    # Get predictions
    result_df = predict(dataset, model)

    # Create a CSV from DataFrame
    response = StreamingResponse(
        io.StringIO(result_df.to_csv(index=False, sep="\t")), media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
