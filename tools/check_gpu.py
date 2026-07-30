import os
import torch

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch version =", torch.__version__)
print("CUDA available =", torch.cuda.is_available())
print("Visible GPU count =", torch.cuda.device_count())

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {torch.cuda.get_device_name(index)} | "
            f"{props.total_memory / 1024**3:.1f} GB"
        )
