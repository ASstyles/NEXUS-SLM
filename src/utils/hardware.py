"""Hardware inspection and capability detection for NEXUS-SLM."""

import os
import platform
import psutil
from typing import Any, Dict


def get_hardware_info() -> Dict[str, Any]:
    """Inspects host machine resources including CPU, RAM, and NVIDIA GPU VRAM."""
    cpu_count = os.cpu_count() or 1
    ram = psutil.virtual_memory()

    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform_machine": platform.machine(),
        "cpu_count": cpu_count,
        "ram_total_gb": round(ram.total / (1024**3), 2),
        "ram_available_gb": round(ram.available / (1024**3), 2),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
        "bitsandbytes_available": False,
        "recommended_precision": "fp32",
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb = round(props.total_memory / (1024**3), 2)
                allocated_gb = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                reserved_gb = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": total_vram_gb,
                    "allocated_vram_gb": allocated_gb,
                    "reserved_vram_gb": reserved_gb,
                    "major": props.major,
                    "minor": props.minor,
                })
            
            # Recommendation based on available VRAM
            primary_vram = info["gpus"][0]["total_vram_gb"] if info["gpus"] else 0
            if primary_vram >= 8.0:
                info["recommended_precision"] = "4bit_qlora_or_fp16"
            elif primary_vram >= 3.5:
                info["recommended_precision"] = "4bit_qlora"
            else:
                info["recommended_precision"] = "8bit_or_cpu"
    except Exception as e:
        info["cuda_error"] = str(e)

    try:
        import bitsandbytes
        info["bitsandbytes_available"] = True
        info["bitsandbytes_version"] = getattr(bitsandbytes, "__version__", "installed")
    except Exception:
        info["bitsandbytes_available"] = False

    return info


def print_hardware_summary() -> None:
    """Pretty prints the hardware diagnostics."""
    info = get_hardware_info()
    print("=" * 60)
    print("NEXUS-SLM Hardware Diagnostics")
    print("=" * 60)
    print(f"OS: {info['os']} ({info['platform_machine']})")
    print(f"CPUs: {info['cpu_count']} cores | RAM: {info['ram_available_gb']} GB / {info['ram_total_gb']} GB")
    if info["cuda_available"]:
        print(f"CUDA: Available (v{info['cuda_version']}) | GPUs: {info['gpu_count']}")
        for gpu in info["gpus"]:
            print(f"  - [{gpu['index']}] {gpu['name']} | VRAM: {gpu['total_vram_gb']} GB")
    else:
        print("CUDA: Not available (Running on CPU)")
    print(f"BitsAndBytes: {'Available' if info['bitsandbytes_available'] else 'Not available'}")
    print(f"Recommended Precision: {info['recommended_precision']}")
    print("=" * 60)


if __name__ == "__main__":
    print_hardware_summary()
