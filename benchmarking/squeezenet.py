import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import psutil
import numpy as np

# Load pretrained SqueezeNet model
model = models.squeezenet1_1(pretrained=True)
model.eval()

# Save model weights to measure size
torch.save(model.state_dict(), "squeezenet1_1.pth")
model_size_mb = os.path.getsize("squeezenet1_1.pth") / (1024 ** 2)
print(f"Model Size (weights only): {model_size_mb:.2f} MB")

# Load test image
image_path = "test.jpg"
image = Image.open(image_path).convert("RGB").resize((224, 224))

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0)

# Measure inference time and CPU
inference_times = []
cpu_samples = []
process = psutil.Process()

start_mem = process.memory_info().rss / (1024 ** 2)  # in MB

with torch.no_grad():
    for _ in range(100):
        cpu_samples.append(process.cpu_percent(interval=None))
        start = time.time()
        _ = model(input_tensor)
        end = time.time()
        inference_times.append((end - start) * 1000)  # ms
        cpu_samples[-1] = process.cpu_percent(interval=None)

end_mem = process.memory_info().rss / (1024 ** 2)
ram_usage = end_mem - start_mem
avg_cpu = np.mean(cpu_samples)

# Stats
times_np = np.array(inference_times)
print(f"\nSqueezeNet Inference Benchmark (100 runs):")
print(f"  Mean Inference Time:      {times_np.mean():.2f} ms")
print(f"  Std Deviation:            {times_np.std():.2f} ms")
print(f"  Min Time:                 {times_np.min():.2f} ms")
print(f"  Max Time:                 {times_np.max():.2f} ms")
print(f"  RAM Usage:                {ram_usage:.2f} MB")
print(f"  CPU Utilization:          {avg_cpu:.2f} %")
print(f"  Model Size (.pth):        {model_size_mb:.2f} MB")
