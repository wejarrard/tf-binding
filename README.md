# tf-binding

This project is focused on trying to predict transcription factor binding sites using deep learning. The project is divided into three phases: preprocessing, pretraining, and finetuning. The preprocessing phase is used to generate the data for pretraining and finetuning. The pretraining and finetuning phases are used to pretrain the model.


## Pretraining

### Preprocessing

Before pretraining, we need to preprocess the data. The preprocessing scripts are located in the `src/preprocessing/` directory. Here are the key scripts:

- `src/preprocessing/negative_generation.py`: This script is used to generate negative samples.
- `src/preprocessing/positive_generation.py`: This script is used to generate positive samples.
- `src/preprocessing/filter.py`: This script is used to filter the generated data.
- `src/preprocessing/test_negative.py`: This script is used to test the generated negative samples.
- `src/preprocessing/test_positive.py`: This script is used to test the generated positive samples.

### Training

After preprocessing and pretraining, we move on to the training phase. The training scripts are located in the `src/training/` directory. Here are the key scripts:

- `src/training/pretrain.py`: This script is used for pretraining the model.
- `src/training/finetune.py`: This script is used for fine-tuning the pretrained model.
- `src/training/deepseq.py`: This script contains the DeepSeq model implementation.
- `src/training/data.py`: This script is used for data handling during training.
- `src/training/config.py`: This script contains the configuration for training.
- `src/training/training_utils.py`: This script contains utility functions for training.


### AWS

Pretraining is the initial phase of training the model. In this project, we use the scripts located in `src/sagemaker/pretrain.ipynb` and `src/sagemaker/pretrain_distributed.ipynb` for pretraining. The former is used for pretraining on a single instance, while the latter is used for pretraining on multiple instances. The scripts are designed to be run on AWS SageMaker.