import time

import boto3

client = boto3.client("sagemaker")

# First, delete the existing model if it exists
try:
    client.delete_model(ModelName="ar-inference")
    print("Existing model deleted successfully.")
except client.exceptions.ClientError as e:
    if e.response["Error"]["Code"] == "ValidationException":
        print("Model does not exist or is already being deleted.")
    else:
        raise  # Re-raise the exception if it's not due to the model not existing

# Now, create the new model with the same name

model_artifacts_s3_url = "s3://tf-binding-sites/finetuning/results/output/AR-22rv1-4k-2024-04-12-03-06-15-583/output/model.tar.gz"

response = client.create_model(
    ModelName="ar-inference",
    ExecutionRoleArn="arn:aws:iam::016114370410:role/service-role/SageMaker-tf-binding-sites",
    PrimaryContainer={
        "Image": "016114370410.dkr.ecr.us-west-2.amazonaws.com/tf-binding-inference:latest",
        "ModelDataUrl": model_artifacts_s3_url,
    },
)

print("New model created successfully.")
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
                "S3Uri": "s3://tf-binding-sites/inference/input/",
            }
        },
        "ContentType": "string",
        # "CompressionType": "Gzip",
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
