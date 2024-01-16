import torch


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        verbose: bool = False,
        delta: float = 0,
        save_path: str = "",
    ):
        self.patience = patience
        self.verbose = verbose
        self.patience_counter = 0
        self.best_loss = float("inf")
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss is None:
            return False

        if val_loss < self.best_loss - self.delta:
            # Improved loss, save the model and reset the counter
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(val_loss, model)
        else:
            # No improvement in loss
            self.patience_counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.patience_counter} out of {self.patience}"
                )

            if self.patience_counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                return True

        return False

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(
                    f"Validation loss decreased to {val_loss:.6f}, model saved to {self.save_path}"
                )

    def state_dict(self):
        """Returns the state of the EarlyStopping instance."""
        return {
            "patience": self.patience,
            "verbose": self.verbose,
            "patience_counter": self.patience_counter,
            "best_loss": self.best_loss,
            "delta": self.delta,
            "save_path": self.save_path,
        }

    def load_state_dict(self, state_dict):
        """Loads the EarlyStopping state."""
        self.patience = state_dict["patience"]
        self.verbose = state_dict["verbose"]
        self.patience_counter = state_dict["patience_counter"]
        self.best_loss = state_dict["best_loss"]
        self.delta = state_dict["delta"]
        self.save_path = state_dict["save_path"]
