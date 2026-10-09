import sys, os, platform, psutil, shutil

print("=== SYSTEM & HARDWARE INSPECTION ===")
print("Python:", sys.version)
print("Platform:", platform.platform())

try:
    import torch
    print("Torch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("GPU Count:", count)
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} | Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is NOT available in this PyTorch build.")
except Exception as e:
    print("Torch error:", e)

ram = psutil.virtual_memory()
print(f"System RAM Total: {ram.total / (1024**3):.2f} GB | Available: {ram.available / (1024**3):.2f} GB")

total_disk, used_disk, free_disk = shutil.disk_usage(".")
print(f"Disk Total: {total_disk / (1024**3):.2f} GB | Free: {free_disk / (1024**3):.2f} GB")

print("\n=== INSTALLED ML DEPENDENCIES ===")
packages = ['transformers', 'peft', 'datasets', 'bitsandbytes', 'accelerate', 'mlflow', 'sklearn', 'huggingface_hub']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")

print("\n=== HUGGING FACE CONNECTIVITY ===")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami() if os.environ.get("HF_TOKEN") else "Anonymous (No HF_TOKEN set)"
    print("HF Auth Status:", who)
except Exception as e:
    print("HF Status:", e)
