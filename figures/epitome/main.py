

from epitome.epitome.models import *
from epitome.epitome.dataset import *

targets = ['FOXA1']
# celltypes = ['K562', 'A549', 'GM12878']

dataset = EpitomeDataset(targets=targets, similarity_targets =['DNase'])

model = EpitomeModel(dataset,
        test_celltypes = ["K562"], # cell line reserved for testing
        max_valid_batches = 1000)

best_model_batches, total_trained_batches, train_valid_losses = model.train(5000,
        patience = 3,
        min_delta = 0.01)

# print("Hello, Docker is working fine!")