import os
import torch
from chop.tools import get_tokenized_dataset
from optuna.samplers import TPESampler
import optuna
import torch.nn as nn
from chop.nn.modules import Identity
from transformers import AutoConfig, AutoModelForSequenceClassification
from chop.tools.utils import deepsetattr
from chop.tools import get_trainer
from chop.pipelines import CompressionPipeline
from chop import MaseGraph
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "DeepWokLab/bert-tiny"
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# Define search space
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
     "linear_layer_choices": ["linear", "identity"],
}

def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)
    for param in ["num_layers", "num_heads", "hidden_size", "intermediate_size"]:
        chosen_value = trial.suggest_categorical(name=param, choices=search_space[param])
        setattr(config, param, chosen_value)

    model = AutoModelForSequenceClassification.from_config(config)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(f"{name}_type", search_space["linear_layer_choices"])
            if new_layer_cls == Identity:
                deepsetattr(model, name, Identity())
    return model

def objective(trial):
    # Quantization and pruning configurations
    quantization_config = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": 16,
                "data_in_frac_width": 8,
                "weight_width": 16,
                "weight_frac_width": 8,
                "bias_width": 16,
                "bias_frac_width": 8,
            }
        },
    }
    pruning_config = {
        "weight": {"sparsity": 0.3, "method": "l1-norm", "scope": "local"},
        "activation": {"sparsity": 0.3, "method": "l1-norm", "scope": "local"},
    }
    
    # Baseline training (Task 1: without compression)
    model = construct_model(trial)
    model.to(device)
    trainer = get_trainer(
        model=model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=2,
    )
    trainer.train()
    baseline_results = trainer.evaluate()
    baseline_acc = baseline_results["eval_accuracy"]

    # Compression pipeline
    mg = MaseGraph(model, hf_input_names=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
    mg.model.to("cpu")
    pipe = CompressionPipeline()
    mg, _ = pipe(
        mg,
        pass_args={
            "quantize_transform_pass": quantization_config,
            "prune_transform_pass": pruning_config,
        },
    )

    # Evaluate after compression without additional training
    model_compressed_no_train = mg.model.to(device)
    trainer_no_train = get_trainer(
        model=model_compressed_no_train,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=0,
    )
    compressed_no_training_acc = trainer_no_train.evaluate()["eval_accuracy"]

    # Post-compression training (1 epoch) then evaluate
    model_compressed_train = mg.model.to(device)
    trainer_post = get_trainer(
        model=model_compressed_train,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=1,
    )
    trainer_post.train()
    compressed_post_training_acc = trainer_post.evaluate()["eval_accuracy"]

    trial.set_user_attr("baseline_acc", baseline_acc)
    trial.set_user_attr("compressed_no_training_acc", compressed_no_training_acc)
    trial.set_user_attr("compressed_post_training_acc", compressed_post_training_acc)

    torch.cuda.empty_cache()
    del model
    return compressed_post_training_acc

if __name__ == "__main__":
    sampler = TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name="bert-tiny-compression-pipeline",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=10,
        timeout=60 * 60 * 24,
        n_jobs=1,
    )

    study.trials_dataframe().to_csv("results_compression.csv")

    baseline_scores = []
    comp_no_train_scores = []
    comp_post_train_scores = []
    for trial in study.trials:
        user_attrs = trial.user_attrs
        baseline_scores.append(user_attrs.get("baseline_acc", 0))
        comp_no_train_scores.append(user_attrs.get("compressed_no_training_acc", 0))
        comp_post_train_scores.append(user_attrs.get("compressed_post_training_acc", 0))

    best_baseline = []
    best_comp_no_train = []
    best_comp_post_train = []
    current_best_baseline = 0
    current_best_comp_no_train = 0
    current_best_comp_post_train = 0

    num_trials = len(study.trials)
    trials_x = list(range(1, num_trials + 1))
    for i in range(num_trials):
        current_best_baseline = max(current_best_baseline, baseline_scores[i])
        current_best_comp_no_train = max(current_best_comp_no_train, comp_no_train_scores[i])
        current_best_comp_post_train = max(current_best_comp_post_train, comp_post_train_scores[i])
        best_baseline.append(current_best_baseline)
        best_comp_no_train.append(current_best_comp_no_train)
        best_comp_post_train.append(current_best_comp_post_train)

    plt.figure(figsize=(10, 6))
    plt.plot(trials_x, best_baseline, marker="o", linestyle="-", label="Baseline (no compression)")
    plt.plot(trials_x, best_comp_no_train, marker="s", linestyle="--", label="Compressed (no post-training)")
    plt.plot(trials_x, best_comp_post_train, marker="^", linestyle="-.", label="Compressed (with post-training)")
    plt.xlabel("Number of trials")
    plt.ylabel("Best cumulative accuracy")
    plt.title("Comparison of compression-aware NAS methods")
    plt.legend()
    plt.grid(True)
    plt.savefig("compression_results.png", dpi=300, bbox_inches="tight")
    plt.show()
