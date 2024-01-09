# tf-binding Project

This project aims to predict transcription factor (TF) binding sites using deep learning. It is divided into three main phases: preprocessing, pretraining, and finetuning.

## Preprocessing Phase

The preprocessing phase prepares the data for both pretraining and finetuning stages.

### Key Scripts

Located in `src/preprocessing/`:

- `negative_generation.py`: Generates negative samples.
- `positive_generation.py`: Generates positive samples.
- `filter.py`: Filters the generated data.
- `test_negative.py`: Tests the generated negative samples.
- `test_positive.py`: Tests the generated positive samples.

## Pretraining Phase

Pretraining is the initial model training phase, conducted on AWS.

### Training Scripts

- Single instance training: `src/sagemaker/pretrain.ipynb`
- Distributed training: `src/sagemaker/pretrain_distributed.ipynb`

These scripts are designed for AWS SageMaker.

## Finetuning Phase

Finetuning involves further training the model with specific data.

### Preprocessing

TODO: Detail the preprocessing steps for finetuning.

### AWS Setup

To run the finetuning script:

1. Navigate to AWS SageMaker in the `us-west-2` region.
2. Go to the Studio tab and select "Open Studio".
3. Locate "TFBinding" under Collaborative Spaces (check permissions if not visible).
4. This opens a Jupyter Notebook-like UI. Select the folder tab.
5. Navigate to `tf-binding/src/sagemaker/tf_finetuning.ipynb`.
6. Run the script. It will initialize instances and start the training process.

### Training Customization

- To train a new model on a different TF, change the data source in the script. For example, replace `s3://tf-binding-sites/pretraining/data/AR_ATAC_broadPeak` with your specific TF data file. Contact Thiady for guidance on generating this file.

### Codebase

- All training code is located in `tf-binding/src/training`.
- The current finetuning script is `tf-prediction.py`.

## TODO List

- [ ] Write detailed README for the pretraining phase.
- [ ] Complete the preprocessing section in the README for finetuning.
- [ ] Add protocols and enhance code readability.
- [ ] Implement the option to run finetuning without pretrained weights.
- [ ] Test the model on novel Transcription Factors (TFs).
- [ ] Attempt predictions on new cell lines.
- [ ] Investigate and resolve issues with LNCAP.
- [ ] Develop a model to predict multiple TFs simultaneously.
- [ ] Create a Hugging Face space using Gradio for the project.
