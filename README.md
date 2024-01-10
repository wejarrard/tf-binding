# TF Binding Prediction

This project is focused on predicting transcription factor (TF) binding sites using deep learning. It's organized into three main phases: preprocessing, pretraining, and finetuning.

### Key Scripts

Located in `src/preprocessing/`:

- **`negative_generation.py`**: Generates negative samples.
- **`positive_generation.py`**: Generates positive samples.
- **`filter.py`**: Filters the generated data.
- **`test_negative.py`**: Tests the generated negative samples.
- **`test_positive.py`**: Tests the generated positive samples.

## Pretraining Phase

Pretraining is the initial model training phase, conducted on AWS.

### Preprocessing

The preprocessing phase prepares the data for the pretraining stage. For additional steps, contact Thiady for the rest of the preprocessing steps to be applied before these scripts in src/preprocessing.

### Training Scripts

- **Single instance training**: `src/sagemaker/pretrain.ipynb`
- **Distributed training**: `src/sagemaker/pretrain_distributed.ipynb`

*Note: These scripts are designed for AWS SageMaker.*

## Finetuning Phase

Finetuning involves further training the model with specific data.

### Preprocessing

Create a bed file with the following columns:
`chr, start, end, cell line, label`

Example:
```
chr1 9978 10253 22Rv1 Positive
chr1 9997 10466 C4-2 Positive
chr1 10001 10433 22Rv1 Positive
chr1 10007 11313 PC-3 Positive
```

*For more details on generating this file, contact Thiady.*

### Training

Steps to run the finetuning script:

1. Navigate to AWS SageMaker in the `us-west-2` region.
2. Go to the Studio tab and select "Open Studio".
3. Locate "TFBinding" under Collaborative Spaces (check permissions if not visible).
4. This opens a Jupyter Notebook-like UI. Select the folder tab.
5. Navigate to `tf-binding/src/sagemaker/tf_finetuning.ipynb`.
6. Run the script to initialize instances and start the training process.

*Note: These scripts are designed for AWS SageMaker.*

### Training Customization

- To train a new model on a different TF, change the data source in the script. For example, replace `s3://tf-binding-sites/pretraining/data/AR_ATAC_broadPeak` with your specific TF data file.

* See Preprocessing for example. Contact Thiady for guidance on generating this file.*

### Codebase

- All training code is located in `tf-binding/src/training`.
- The current finetuning script is `tf-prediction.py`.

## TODO

- [x] Complete the preprocessing section in the README for finetuning.
- [ ] Write detailed README for the pretraining phase.
- [ ] Add protocols and enhance code readability.
- [ ] Investigate and resolve issues with LNCAP.
- [ ] Implement the option to run finetuning without pretrained weights.
- [ ] Test the model on novel Transcription Factors.
- [ ] Attempt predictions on new cell lines.
- [ ] Develop a model to predict multiple TFs simultaneously.
- [ ] Create a Hugging Face space using Gradio for the project.
