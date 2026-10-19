"""Public GPU specs, used only to ground LLM prompts."""
from __future__ import annotations

GPU_SPECS: dict[str, dict[str, str]] = {
    "RTX-4090": {"GPU Architecture": "Ada Lovelace", "GPU Memory": "24GB GDDR6X",
                 "Memory Bandwidth": "1008 GB/s", "FP32 TFLOPS": "82.6",
                 "TF32 Tensor Core TFLOPS": "82.6", "FP16 Tensor Core TFLOPS": "165.2",
                 "Shared memory capacity per SM": "100 KB", "Register File Size": "64K 32-bit registers per SM"},
    "A100": {"GPU Architecture": "Ampere", "GPU Memory": "40GB HBM2e",
             "Memory Bandwidth": "1555 GB/s", "FP32 TFLOPS": "19.5",
             "TF32 Tensor Core TFLOPS": "156", "FP16 Tensor Core TFLOPS": "312",
             "Shared memory capacity per SM": "164 KB", "Register File Size": "64K 32-bit registers per SM"},
    "H100": {"GPU Architecture": "Hopper", "GPU Memory": "80GB HBM3",
             "Memory Bandwidth": "3350 GB/s", "FP32 TFLOPS": "67",
             "TF32 Tensor Core TFLOPS": "495", "FP16 Tensor Core TFLOPS": "989",
             "Shared memory capacity per SM": "228 KB", "Register File Size": "64K 32-bit registers per SM"},
    "L40S": {"GPU Architecture": "Ada Lovelace", "GPU Memory": "48GB GDDR6",
             "Memory Bandwidth": "864 GB/s", "FP32 TFLOPS": "91.6",
             "TF32 Tensor Core TFLOPS": "183", "FP16 Tensor Core TFLOPS": "362",
             "Shared memory capacity per SM": "100 KB", "Register File Size": "64K 32-bit registers per SM"},
}


def gpu_block(name: str) -> str:
    info = GPU_SPECS.get(name, {})
    lines = [f"GPU Name: {name}", f"Architecture: {info.get('GPU Architecture', 'Unknown')}"]
    lines += [f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"]
    return "\n".join(lines)
