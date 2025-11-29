import os, subprocess
import torch
from PIL import Image
import numpy as np

def get_free_gpus(mem_threshold=5000, util_threshold=10):
    # Query GPU memory used and GPU utilization using nvidia-smi
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        free_gpus = []
        # Each line has the format: "<memory_used>, <utilization>"
        for i, line in enumerate(result.strip().split("\n")):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                mem_used = int(parts[0])
                util = int(parts[1])
                if mem_used < mem_threshold and util < util_threshold:
                    free_gpus.append(i)
        if not free_gpus:
            print("No free GPUs available. Terminating process.")
            exit(1)
        return free_gpus
    except Exception as e:
        print("Error running nvidia-smi:", e)
        return []
    
def load_image(image_path):
    def process_single_image(img_path):
        img = Image.open(img_path).convert("RGB")
        x = torch.from_numpy(np.array(img)).to(torch.float32)
        x = x.permute(2, 0, 1)
        return (x - 128) / 127.5
    
    if os.path.isfile(image_path):
        return process_single_image(image_path)
    
    elif os.path.isdir(image_path):
        images = [os.path.join(image_path, f) \
            for f in sorted(os.listdir(image_path)) if f.endswith('.png')]
        processed_images = [process_single_image(img) for img in images]
        return torch.stack(processed_images, dim=0)
    pass

def save_image(tensor, path, skip=False):
    if skip and os.path.exists(path):
        return
    size = tensor.shape[-1]
    image = (tensor.cpu() * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(-1, 3, size, size)
    image = image.permute(0, 2, 3, 1).reshape(size, size, 3).numpy()
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    Image.fromarray(image, 'RGB').save(path)
    pass
# =====================================================================================
