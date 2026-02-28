# CUDA Setup & GPU Acceleration

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows software to use the GPU for general-purpose computation. In the context of ECHO-TOON, CUDA dramatically accelerates:

- YOLOv8 inference (character and bubble detection)
- EasyOCR text recognition
- Dia TTS model inference
- BLIP2 / CLIP scene description
- Sentence Transformer embeddings
- Any PyTorch model

Without CUDA, all of these run on CPU — which is 10x–100x slower depending on the model.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ |
| VRAM | 6 GB | 8–12 GB |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB free |
| OS | Windows 10/11 | Windows 11 |

> AMD GPUs are **not supported** by CUDA. Use ROCm (Linux only) for AMD, or run CPU mode.

---

## Step 1 — Check Your GPU

Open PowerShell or Command Prompt:

```powershell
nvidia-smi
```

Expected output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 537.xx    Driver Version: 537.xx    CUDA Version: 12.x          |
|-------------------------------|------------------|--------------------------|
| GPU  Name          TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce ...  WDDM | 00000000:01:00.0  On |                  N/A |
```

If this command fails, your NVIDIA drivers are not installed.

---

## Step 2 — Install NVIDIA Drivers

1. Go to: https://www.nvidia.com/Download/index.aspx
2. Select your GPU model and OS
3. Download and install the **Game Ready** or **Studio** driver
4. Restart your PC after installation
5. Verify with `nvidia-smi`

---

## Step 3 — Install CUDA Toolkit

### Check what CUDA version your driver supports

```powershell
nvidia-smi
```

Look at the top-right corner: `CUDA Version: 12.x` — this is the **maximum** CUDA version your driver supports.

### Download CUDA Toolkit

Go to: https://developer.nvidia.com/cuda-toolkit-archive

For this project we use **CUDA 11.8** (compatible with PyTorch cu118 builds):

1. Select: CUDA Toolkit 11.8.0
2. OS: Windows → x86_64 → 11 → exe (local)
3. Download and run the installer
4. Choose **Custom Install** → uncheck GeForce Experience if already installed

### Verify CUDA installation

```powershell
nvcc --version
```

Expected output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Built on ...
Cuda compilation tools, release 11.8, V11.8.xxx
```

---

## Step 4 — Install cuDNN

cuDNN (CUDA Deep Neural Network library) accelerates deep learning frameworks.

1. Go to: https://developer.nvidia.com/cudnn
2. Create a free NVIDIA developer account if needed
3. Download **cuDNN 8.x for CUDA 11.8**
4. Extract the zip file
5. Copy the contents into your CUDA installation directory:

```
cuDNN extracted folder:
  bin\          → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\
  include\      → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include\
  lib\x64\      → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64\
```

---

## Step 5 — Add CUDA to System PATH

Open: Control Panel → System → Advanced System Settings → Environment Variables

Under **System Variables**, find `Path` and add:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

Restart your terminal after this.

---

## Step 6 — Install PyTorch with CUDA Support

This project uses **PyTorch 2.x with CUDA 11.8**.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Do NOT use `pip install torch` alone — that installs the CPU-only version.

### Verify PyTorch CUDA

```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.device_count())      # 1 (or more)
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX XXXX
print(torch.__version__)              # 2.x.x+cu118
```

Or run the project's existing check:

```bash
python Text_To_Speech_with_dia/GPU_test.py
```

---

## Step 7 — Verify Full Stack

```python
import torch
from ultralytics import YOLO
import easyocr

# PyTorch CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# YOLOv8 on GPU
model = YOLO('yolov8n.pt')
results = model('test_image.jpg', device='cuda')

# EasyOCR on GPU
reader = easyocr.Reader(['en'], gpu=True)
```

---

## CUDA Versions Reference Table

| PyTorch Version | CUDA Version | Install Command |
|---|---|---|
| 2.7.x | CUDA 12.8 | `--index-url .../whl/cu128` |
| 2.7.x | CUDA 12.6 | `--index-url .../whl/cu126` |
| 2.7.x | CUDA 11.8 | `--index-url .../whl/cu118` |
| 2.x.x | CPU only | `pip install torch` |

Always match the CUDA version between your toolkit and PyTorch build.

---

## Common Issues

### `torch.cuda.is_available()` returns `False`

1. Wrong PyTorch build — reinstall with the `cu118` index URL
2. Driver too old — update NVIDIA drivers
3. CUDA toolkit not found — check PATH variables
4. Virtual environment issue — activate venv before installing

### `CUDA out of memory`

```python
# Free cache between inferences
torch.cuda.empty_cache()

# Use half precision to reduce VRAM usage
model = model.half()
```

### `nvcc not found`

CUDA toolkit is not in PATH. Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` to System PATH.

### `DLL load failed` on Windows

Install the **Microsoft Visual C++ Redistributable** (required by CUDA on Windows):
https://aka.ms/vs/17/release/vc_redist.x64.exe

---

## Running Without GPU (CPU Mode)

All modules in ECHO-TOON fall back to CPU if CUDA is unavailable. Simply ensure models are loaded without device specifiers or with `device='cpu'`:

```python
# EasyOCR CPU mode
reader = easyocr.Reader(['en'], gpu=False)

# YOLOv8 CPU mode
model = YOLO('yolov8n.pt')
results = model('image.jpg', device='cpu')

# PyTorch CPU
device = torch.device('cpu')
```

CPU inference is significantly slower but fully functional for testing.

---

## GPU Memory Estimates

| Model | VRAM Required |
|---|---|
| YOLOv8n | ~0.5 GB |
| EasyOCR | ~0.8 GB |
| Dia 1.6B (float16) | ~3.5 GB |
| BLIP2 | ~5–8 GB |
| LLaMA3 7B (quantized Q4) | ~4–5 GB |
| All combined (streaming) | ~8–12 GB |

For 6 GB VRAM cards, run models sequentially (unload between steps) rather than all at once.
