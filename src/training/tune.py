import json

import mlflow
import optuna
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import build_dataloaders
from src.evaluation.evaluate import predict
from src.models.resnet import VisionIQResNet, get_device
from src.training.train import train_one_epoch
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main():
    config = load_config()
    device = get_device()
    tcfg = config["tuning"]
    train_dl, val_dl, _ = build_dataloaders(config, "standard")
    train_small = Subset(train_dl.dataset, range(tcfg["train_subset"]))
    val_small = DataLoader(Subset(val_dl.dataset, range(tcfg["val_subset"])), batch_size=128)

    (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    def objective(trial):
        # 1. Optuna proposes a combination of hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        }
        # 2. short training run with those settings
        set_seed(config["project"]["seed"])
        model = VisionIQResNet(dropout=params["dropout"],
                               input_size=config["data"]["image_size"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"],
                                      weight_decay=params["weight_decay"])
        loader = DataLoader(train_small, batch_size=params["batch_size"], shuffle=True)
        train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, device)
        # 3. score = validation macro F1
        y_true, y_pred, _ = predict(model, val_small, device)
        f1 = f1_score(y_true, y_pred, average="macro")
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("validation_f1", f1)
        logger.info("trial %d: %s -> val F1 %.4f", trial.number, params, f1)
        return f1

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config["project"]["seed"]))
    with mlflow.start_run(run_name="optuna_study"):
        study.optimize(objective, n_trials=tcfg["n_trials"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_validation_f1", study.best_value)

    out = PROJECT_ROOT / "reports" / "best_params.json"
    out.write_text(json.dumps(study.best_params, indent=2))
    logger.info("best trial #%d F1=%.4f params=%s -> %s",
                study.best_trial.number, study.best_value, study.best_params, out)


if __name__ == "__main__":
    main()