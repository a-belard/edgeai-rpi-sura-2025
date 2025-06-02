import time
import torch
import numpy as np
import psutil
import os
from torchvision import models, transforms
from PIL import Image

# Set PyTorch backend for quantized models
torch.backends.quantized.engine = 'qnnpack'

# Load quantized MobileNetV2
model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()

#saving the model to get the size
torch.save(model.state_dict(), "mobilenetv2_quantized.pth")
model_size_mb = os.path.getsize("mobilenetv2_quantized.pth") / (1024 ** 2)
print(f"Model Size: {model_size_mb:.2f} MB")

# Load image and preprocess
image_path = "test.jpg"  # Replace with your image
image = Image.open(image_path).convert("RGB").resize((224, 224))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0)

# Benchmark
inference_times = []
cpu_percentages = []

process = psutil.Process(os.getpid())
start_mem = process.memory_info().rss / (1024 ** 2)  # MB

with torch.no_grad():
    for _ in range(100):
        psutil.cpu_percent(interval=None)  # Reset CPU percent reading
        start_time = time.time()
        _ = model(input_tensor)
        end_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)

        inference_times.append((end_time - start_time) * 1000)  # in ms
        cpu_percentages.append(cpu_usage)

end_mem = process.memory_info().rss / (1024 ** 2)
ram_used_mb = end_mem - start_mem

# Stats
times_np = np.array(inference_times)
cpu_np = np.array(cpu_percentages)

print("\n===== MobileNetV2 Quantized Benchmark =====")
print(f"Model Size:            {model_size_mb:.2f} MB")
print(f"Mean Inference Time:   {times_np.mean():.2f} ms")
print(f"Std Deviation:         {times_np.std():.2f} ms")
print(f"Min Time:              {times_np.min():.2f} ms")
print(f"Max Time:              {times_np.max():.2f} ms")
print(f"RAM Usage:             {ram_used_mb:.2f} MB")
print(f"CPU Utilization:       {cpu_np.mean():.2f} %")
print("===========================================")
