import io
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import uvicorn
from dataloader import EnhancedTFRecordDataset
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from inference import load_model, predict

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


@app.post("/invocations")
async def get_inference(file: UploadFile):
    # Load your model (consider loading it outside of request handling if it's heavy)
    # model = load_model("/opt/ml/model")
    if os.getenv("MODEL_PATH"):
        model_path = os.getenv("MODEL_PATH")
    else:
        model_path = "data/"
    model = load_model(model_path)

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
    }
    dataset = EnhancedTFRecordDataset(
        data_path, index_path, description, compression_type="gzip"
    )

    # Get predictions
    result_df = predict(dataset, model)

    # Create a CSV from DataFrame
    response = StreamingResponse(
        io.StringIO(result_df.to_csv(index=False)), media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
