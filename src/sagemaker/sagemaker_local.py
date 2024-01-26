# Single GPU
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.local import LocalSession

role = 'arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001'
sess = LocalSession()
sess.config = {'local': {'local_code': True}}

output_s3_path = "file://./output/"


use_spot_instances = True
max_wait = 1209600 if use_spot_instances else None
checkpoint_s3_bucket="s3://tf-binding-sites/finetuning/results/checkpointing"
checkpoint_local_path="/opt/ml/checkpoints"

estimator = PyTorch(
    base_job_name="tf-finetuning",
    entry_point="tf_prediction.py",
    source_dir="../training",
    # output_path=output_s3_path,
    role=role,
    py_version="py310",
    framework_version='2.0.0',
    instance_count=1,
    instance_type='local',
    hyperparameters={
        'learning-rate': 1e-5
    },
    # tensorboard_output_config=tensorboard_output_config,
    # use_spot_instances=use_spot_instances,
    # max_wait=max_wait,
    # checkpoint_s3_uri=checkpoint_s3_bucket,
    # checkpoint_local_path=checkpoint_local_path
)

estimator.fit({'training': 'file://../training/data/'})

