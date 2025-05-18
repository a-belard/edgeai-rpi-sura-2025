import time
import torch
import numpy as np
import psutil
from torchvision import models, transforms
from PIL import Image

# Set PyTorch quantized backend
torch.backends.quantized.engine = 'qnnpack'

# Load quantized MobileNetV2
model = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model.eval()
model = torch.jit.script(model)

# Load image
image_path = "test.jpg" 
image = Image.open(image_path).convert("RGB").resize((224, 224))

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# Run inference 100 times
times = []
with torch.no_grad():
    for _ in range(100):
        start = time.time()
        output = model(input_tensor)
        end = time.time()
        times.append((end - start) * 1000)  # ms

# Memory usage
ram_usage = psutil.Process().memory_info().rss / 1024 / 1024  # in MB

# Stats
times_np = np.array(times)
print(f"Inference Benchmark over 100 runs:")
print(f"  Mean Inference Time: {times_np.mean():.2f} ms")
print(f"  Std Deviation:       {times_np.std():.2f} ms")
print(f"  Min Time:            {times_np.min():.2f} ms")
print(f"  Max Time:            {times_np.max():.2f} ms")
print(f"  RAM Usage:           {ram_usage:.2f} MB")
