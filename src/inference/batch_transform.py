import time

import boto3

client = boto3.client("sagemaker")

response = client.create_transform_job(
    TransformJobName=f"ar-inference-{int(time.time())}",
    ModelName="ar-inference",
    MaxConcurrentTransforms=1,
    MaxPayloadInMB=100,
    BatchStrategy="MultiRecord",  #
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://tf-binding-sites/inference/input/dataset.tfrecord",
            }
        },
        "ContentType": "string",
        "CompressionType": "None",
        "SplitType": "TFRecord",
    },
    TransformOutput={
        "S3OutputPath": "s3://tf-binding-sites/inference/output",
        "Accept": "string",
        "AssembleWith": "Line",
    },
    DataCaptureConfig={
        "DestinationS3Uri": "s3://tf-binding-sites/inference/output",
    },
    TransformResources={
        "InstanceType": "ml.g4dn.xlarge",
        "InstanceCount": 1,
    },
)
