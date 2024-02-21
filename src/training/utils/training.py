import os

from enformer_pytorch import Enformer
from transformers import PreTrainedModel


def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in named_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def count_directories(path: str) -> int:
    # Check if path exists and is a directory
    assert os.path.exists(path), "The specified path does not exist."
    assert os.path.isdir(path), "The specified path is not a directory."

    # Count only directories within the specified path
    directory_count = sum(
        os.path.isdir(os.path.join(path, i)) for i in os.listdir(path)
    )
    return directory_count


def transfer_enformer_weights_to_(
    model: PreTrainedModel, transformer_only: bool = False
) -> PreTrainedModel:
    # Load pretrained weights
    enformer = Enformer.from_pretrained("EleutherAI/enformer-official-rough")

    # print(enformer)

    if transformer_only:
        # Specify components to transfer
        components_to_transfer = ["transformer"]

        # Initialize an empty state dict
        state_dict_to_load = {}

        # Iterate over each component
        for component in components_to_transfer:
            # Extract and add weights of the current component to the state dict
            component_dict = {
                key: value
                for key, value in enformer.state_dict().items()
                if key.startswith(component)
            }

            # Check if the component dict is not empty
            if component_dict:
                state_dict_to_load.update(component_dict)
                # Print confirmation if weights were transferred
                print(f"Weights successfully transferred for component: {component}")
            else:
                # Print a message if no weights were found for the component
                print(f"No weights to transfer for component: {component}")
    else:
        pass

    return model
