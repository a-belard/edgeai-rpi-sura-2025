import torch
import torchvision.models as models
import time
import psutil
import os

# Load pre-trained SqueezeNet model
model = models.squeezenet1_1(pretrained=True)
model.eval()

# Device setup
device = torch.device("cpu")
model.to(device)

# Dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Benchmarking without warm-up
runs = 100
inference_times = []

print(f"Benchmarking SqueezeNet for {runs} runs (no warm-up)...\n")
for i in range(runs):
    start_time = time.time()
    with torch.no_grad():
        _ = model(dummy_input)
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"Run {i+1:03}: {inference_time:.4f} seconds")

# Results
avg_time = sum(inference_times) / runs
std_dev = (sum([(x - avg_time) ** 2 for x in inference_times]) / runs) ** 0.5
ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

print("\n--- Final Benchmark Results ---")
print(f"Average Inference Time: {avg_time:.4f} seconds")
print(f"Standard Deviation: {std_dev:.4f} seconds")
print(f"RAM Usage: {ram_usage:.2f} MB")
