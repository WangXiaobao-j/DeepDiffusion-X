"""ANN architecture definition and loading of the trained model and scaler."""
import pickle
import torch
import torch.nn as nn

from . import descriptor_names


class ANNModel(nn.Module):
    def __init__(self, input_size=13, h1=128, h2=128, h3=64,
                 dropout=0.1, use_batchnorm=True, activation="relu"):
        super().__init__()
        self.hidden1 = nn.Linear(input_size, h1)
        self.bn1 = nn.BatchNorm1d(h1) if use_batchnorm else nn.Identity()
        self.hidden2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2) if use_batchnorm else nn.Identity()
        self.hidden3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3) if use_batchnorm else nn.Identity()
        self.output = nn.Linear(h3, 1)
        self.act = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.bn1(self.hidden1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.hidden2(x)))
        x = self.dropout(x)
        x = self.act(self.bn3(self.hidden3(x)))
        x = self.dropout(x)
        return self.output(x)


def _make_activation(name: str):
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
    }.get(name, nn.ReLU())


def resolve_feature_names(feature_names) -> list:
    """
    Map the feature names stored in a checkpoint to the internal column keys
    used by the descriptor pipeline, preserving order. Names already given as
    internal keys, and names with no known alias, are returned unchanged.
    """
    resolved = []
    for name in feature_names:
        if name in descriptor_names.DESCRIPTORS:
            resolved.append(name)
        else:
            resolved.append(descriptor_names.resolve(name) or name)
    return resolved


def load_ann_checkpoint(pth_path: str, device="cpu", activation="relu", log=None):
    """
    Load the checkpoint and return (model, feature_names, hyperparams).
    Feature names are resolved to internal column keys; the order, and
    therefore the column order expected by the scaler, is preserved.
    The returned model is in eval() mode with strictly matched weights.
    """
    ckpt = torch.load(pth_path, map_location=device, weights_only=False)
    hp = ckpt["hyperparams"]
    stored_names = list(ckpt["feature_names"])
    feature_names = resolve_feature_names(stored_names)
    if log is not None:
        mapped = [(s, r) for s, r in zip(stored_names, feature_names) if s != r]
        if mapped:
            log("  Checkpoint feature names mapped to internal keys: "
                + ", ".join(f"{s} -> {r}" for s, r in mapped))
        unknown = [r for r in feature_names if r not in descriptor_names.DESCRIPTORS]
        if unknown:
            log(f"  Warning: unrecognized feature name(s) in checkpoint: {unknown}")

    model = ANNModel(
        input_size=len(feature_names),
        h1=hp["hidden_size1"], h2=hp["hidden_size2"], h3=hp["hidden_size3"],
        dropout=hp.get("dropout", 0.1),
        use_batchnorm=hp.get("use_batchnorm", True),
        activation=activation,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, feature_names, hp


def load_scaler(pkl_path: str):
    """
    Load the fitted scaler. The training script saves it with
    joblib.dump(), which wraps numpy arrays in its own container format
    to support memory mapping -- plain pickle.load() on such a file
    raises "invalid load key" errors. joblib.load() is required; a
    plain-pickle fallback is kept for the rare case of a file saved
    with pickle.dump() directly.
    """
    try:
        import joblib
        return joblib.load(pkl_path)
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            f"joblib.load() failed to read {pkl_path}: {exc}\n"
            f"Confirm this is the original file produced by "
            f"config.SCALER_SAVE_FILE in the training script, and that "
            f"it was not truncated or copied in text mode."
        ) from exc

    with open(pkl_path, "rb") as f:
        return pickle.load(f)
