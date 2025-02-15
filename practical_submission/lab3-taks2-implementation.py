import torch
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.nn.quantized.modules.linear import (
    LinearInteger, LinearMinifloatDenorm, LinearMinifloatIEEE, LinearLog,
    LinearBlockFP, LinearBlockMinifloat, LinearBlockLog, LinearBinary,
    LinearBinaryScaling, LinearBinaryResidualSign
)
from chop.tools import get_trainer
from transformers import AutoModelForSequenceClassification

checkpoint = "prajjwal1/bert-tiny"
base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from chop.tools import get_tokenized_dataset
dataset_name = "imdb"
tokenizer_checkpoint = "bert-base-uncased"
dataset, tokenizer = get_tokenized_dataset(dataset_name, tokenizer_checkpoint, return_tokenizer=True)

bitwidth_levels = [8, 16, 32]
quantization_schemes = [
    "integer", "binary", "ternary", "minifloat_ieee", "minifloat_denorm",
    "log", "block_fp", "block_minifloat", "block_log"
]

def construct_model(trial):
    """Construct a model with a uniform precision setting for all layers."""
    trial_model = deepcopy(base_model)
    
    bitwidth = trial.suggest_categorical("bitwidth", bitwidth_levels)
    quant_scheme = trial.suggest_categorical("quantization_scheme", quantization_schemes)
    
    # Global configuration dictionary
    config = {
        "name": quant_scheme,
        "weight_width": bitwidth,
        "weight_frac_width": trial.suggest_categorical("weight_frac_width", [2, 4, 8]),
        "data_in_width": bitwidth,
        "data_in_frac_width": trial.suggest_categorical("data_in_frac_width", [2, 4, 8]),
        "bias_width": bitwidth,
        "bias_frac_width": trial.suggest_categorical("bias_frac_width", [2, 4, 8]),
    }
    
    # Additional parameters for specific quantization schemes
    if quant_scheme in ["minifloat_ieee", "minifloat_denorm"]:
        config.update({
            "exponent_width": trial.suggest_categorical("exponent_width", [2, 4, 6]),
            "mantissa_bits": trial.suggest_categorical("mantissa_bits", [3, 5, 7]),
        })
    elif quant_scheme in ["log"]:
        config.update({
            "log_base": trial.suggest_categorical("log_base", [2, 10]),
        })
    elif quant_scheme in ["block_fp", "block_minifloat", "block_log"]:
        config.update({
            "block_size": trial.suggest_categorical("block_size", [16, 32, 64])
        })
    elif quant_scheme in ["binary", "ternary"]:
        config.update({
            "scaling_factor": trial.suggest_float("scaling_factor", 0.1, 2.0)
        })
    
    # Apply quantization to all Linear layers in the model
    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer = LinearInteger(
                in_features=layer.in_features,
                out_features=layer.out_features,
                config=config
            )
            new_layer.weight.data = layer.weight.data
            deepsetattr(trial_model, name, new_layer)
    
    return trial_model, bitwidth

def objective(trial):
    """Optuna objective function: trains and evaluates the model."""
    model, bitwidth = construct_model(trial)
    
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    
    trial.set_user_attr("bitwidth", bitwidth)
    trial.set_user_attr("accuracy", eval_results["eval_accuracy"])
    
    return eval_results["eval_accuracy"]

n_trials = 10
study = optuna.create_study(direction="maximize")

with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
    def wrapped_objective(trial):
        result = objective(trial)
        pbar.update(1)
        return result
    study.optimize(wrapped_objective, n_trials=n_trials)


total_trials = len(study.trials)
curves = {scheme: [] for scheme in quantization_schemes}
current_best = {scheme: float("-inf") for scheme in quantization_schemes}

sorted_trials = sorted(study.trials, key=lambda t: t.number)

for trial in sorted_trials:
    if trial.value is not None:
        precision = trial.params["quantization_scheme"]
        accuracy = trial.value
        
        current_best[precision] = max(current_best[precision], accuracy)
    
    for scheme in quantization_schemes:
        curves[scheme].append(current_best[scheme])

precision_trial_counts = {scheme: 0 for scheme in quantization_schemes}
for trial in sorted_trials:
    if trial.value is not None:
        precision = trial.params["quantization_scheme"]
        precision_trial_counts[precision] += 1

plt.figure(figsize=(10, 6))
for scheme, values in curves.items():
    if max(values) > float("-inf"):
        plt.plot(range(1, total_trials + 1), values, marker='o', linestyle='-', label=scheme)

plt.xlabel("Trial Number")
plt.ylabel("Maximum Achieved Accuracy")
plt.title("Optuna Search Progression per Quantization Scheme")
plt.legend()
plt.grid(True)

plt.savefig("lab3-taks2-result.png")