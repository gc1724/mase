import torch
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
from chop.tools.utils import deepsetattr
from copy import deepcopy
from chop.nn.quantized.modules.linear import LinearInteger
from chop.tools import get_trainer
from transformers import AutoModelForSequenceClassification

checkpoint = "prajjwal1/bert-tiny"
base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from chop.tools import get_tokenized_dataset
dataset_name = "imdb"
tokenizer_checkpoint = "bert-base-uncased"
dataset, tokenizer = get_tokenized_dataset(dataset_name, tokenizer_checkpoint, return_tokenizer=True)

# Define the search space
search_space = {
    "linear_layer_choices": ["torch.nn.Linear", "LinearInteger"],
}

def construct_model(trial):
    """Builds a model with dynamically assigned precision for each layer."""
    trial_model = deepcopy(base_model)

    for name, layer in trial_model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            new_layer_str = trial.suggest_categorical(
                f"{name}_type",
                search_space["linear_layer_choices"],
            )

            if new_layer_str == "torch.nn.Linear":
                continue
            elif new_layer_str == "LinearInteger":
                new_layer_cls = LinearInteger

            kwargs = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
            }

            # Assign precision settings dynamically for IntegerLinear layers
            if new_layer_str == "LinearInteger":
                kwargs["config"] = {
                    "data_in_width": trial.suggest_categorical(f"{name}_data_in_width", [8, 16, 32]),
                    "data_in_frac_width": trial.suggest_categorical(f"{name}_data_in_frac_width", [2, 4, 8]),
                    "weight_width": trial.suggest_categorical(f"{name}_weight_width", [8, 16, 32]),
                    "weight_frac_width": trial.suggest_categorical(f"{name}_weight_frac_width", [2, 4, 8]),
                    "bias_width": trial.suggest_categorical(f"{name}_bias_width", [8, 16, 32]),
                    "bias_frac_width": trial.suggest_categorical(f"{name}_bias_frac_width", [2, 4, 8]),
                }

            # Replace the original layer with the new quantized layer
            new_layer = new_layer_cls(**kwargs)
            new_layer.weight.data = layer.weight.data
            deepsetattr(trial_model, name, new_layer)

    return trial_model

def objective(trial):
    """Optuna objective function: trains and evaluates the model."""
    model = construct_model(trial)

    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )

    trainer.train()
    eval_results = trainer.evaluate()

    trial.set_user_attr("model", model)

    return eval_results["eval_accuracy"]

n_trials = 10
study = optuna.create_study(direction="maximize")

with tqdm(total=n_trials, desc="Optuna Trials") as pbar:
    def wrapped_objective(trial):
        result = objective(trial)
        pbar.update(1)
        return result

    study.optimize(wrapped_objective, n_trials=n_trials)

best_values = []
best_so_far = float("-inf")

for i, trial in enumerate(study.trials):
    if trial.value is not None:
        if i == 0:
            best_values.append(trial.value)
            best_so_far = trial.value
        else:
            best_so_far = max(best_so_far, trial.value)
            best_values.append(best_so_far)

print(f"Best accuracy progression: {best_values}")

print("\nFinal Optuna results:")
for trial in study.trials:
    print(f"Trial {trial.number} - Accuracy: {trial.value} - Params: {trial.params}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(best_values)+1), best_values, marker='o', linestyle='-')
plt.xlabel("Number of Trials")
plt.ylabel("Best Accuracy Achieved")
plt.title("Optuna Search Progression")
plt.grid(True)

plt.savefig("lab3-taks1-result.png")