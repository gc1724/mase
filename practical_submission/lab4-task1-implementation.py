import torch
import time
from chop.models import get_model
from chop.dataset import get_dataset_info

def timed_gpu(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def timed_cpu(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start

def get_data(batch_size=128):
    return torch.randn(batch_size, 3, 224, 224)

def time_model(fn, n=1000, device="cpu", warmup=0):
    times = []
    data = get_data().to(device)
    # Warmup iterations
    for _ in range(warmup):
        if device == "cpu":
            _ = fn(data.cpu())
        else:
            _ = fn(data)
    # Timed iterations
    for _ in range(n):
        if device == "cpu":
            _, t = timed_cpu(lambda: fn(data.cpu()))
        else:
            _, t = timed_gpu(lambda: fn(data))
        times.append(t)
    return sum(times) / len(times)

def evaluate_model(model, device="cpu", n=1000, warmup=0):
    model.to(device)
    return time_model(model, n=n, device=device, warmup=warmup)

# Function to run experiment for a given model name, iteration count, warmup, and device
def run_experiment(model_name, iterations, warmup, device):
    dataset_info = get_dataset_info("imagenet")
    # Load baseline model
    model = get_model(model_name, pretrained=True, dataset_info=dataset_info)
    # Compile model
    compiled_model = torch.compile(model)
    model.to(device)
    compiled_model.to(device)
    baseline_time = time_model(model, n=iterations, device=device, warmup=warmup)
    compiled_time = time_model(compiled_model, n=iterations, device=device, warmup=warmup)
    print(f"{model_name} on {device} with {iterations} iterations and {warmup} warmup:")
    print(f"baseline: {baseline_time:.4f} s, compiled: {compiled_time:.4f} s")
    if compiled_time < baseline_time:
        print("speedup observed")
    else:
        print("no speedup")
    print("")

if __name__ == "__main__":
    iteration_list = [5, 50, 100]
    warmup_list = [0, 10]  # no warmup and with warmup
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:1")
    for device in devices:
        for model_name in ["resnet18", "resnet50"]:
            for n in iteration_list:
                for warm in warmup_list:
                    run_experiment(model_name, n, warm, device)
