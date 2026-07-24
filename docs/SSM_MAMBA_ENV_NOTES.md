# StreamVAD Stage 1 SSM/Mamba 环境配置记录

日期：2026-07-23

## 背景

StreamMind 的 EPFE 使用 Mamba/SSM 模块。Stage 1 LoRA smoke 接入时，模型路径会导入：

```text
StreamMind/streammind/model/multimodal_projector/ssm.py
```

其中依赖：

```python
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
```

因此只要使用 `--mm_projector_type mamba`，就必须正确安装 `mamba-ssm` 及其 CUDA extension。

## 已确认的基础环境

服务器当前使用：

```text
Python: 3.10
torch: 2.6.0+cu124
CUDA runtime: 12.4
CUDA toolkit: /usr/local/cuda-12.4
nvcc: /usr/local/cuda-12.4/bin/nvcc
```

验证命令：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY

/usr/local/cuda-12.4/bin/nvcc --version
```

## 遇到的问题与解决方式

### 1. 裸 `pip` 指向系统环境

现象：

```text
Using pip ... from /usr/lib/python3/dist-packages/pip
Defaulting to user installation
ModuleNotFoundError: No module named 'torch'
```

原因：

`which python` 指向 conda 环境，但 `which pip` 指向 `/bin/pip`，导致安装包进入系统/用户环境，而不是 `streamvad` conda 环境。

解决：

后续安装统一使用：

```bash
python -m pip install ...
```

不要直接使用裸 `pip`。

### 2. torch / torchvision 版本被改坏

现象：

```text
torchvision 0.21.0+cu124 requires torch==2.6.0, but you have torch 2.13.0
```

原因：

安装其他包时依赖解析把 torch 升级/替换成了不匹配版本。

解决：

重新安装 PyTorch CUDA 12.4 组合：

```bash
python -m pip uninstall -y torch torchvision torchaudio

python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

注意版本号不写 `+cu124`，CUDA variant 由 `--index-url` 决定。

### 3. `TRANSFORMERS_CACHE` 导入失败

现象：

```text
cannot import name 'TRANSFORMERS_CACHE' from 'transformers'
```

原因：

StreamMind 老代码使用：

```python
from transformers import TRANSFORMERS_CACHE
```

新版 `transformers` 已不再从顶层暴露该变量。

解决：

降级 `transformers`：

```bash
python -m pip install transformers==4.37.2 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. `peft` 与旧 transformers 不兼容

现象：

```text
cannot import name 'EncoderDecoderCache' from 'transformers'
```

原因：

较新的 `peft` 依赖较新的 `transformers` API，但 StreamMind 需要较旧的 `transformers==4.37.2`。

解决：

降级 `peft`：

```bash
python -m pip uninstall -y peft
python -m pip install peft==0.10.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 没有 `nvcc`

现象：

```text
Command 'nvcc' not found
```

原因：

服务器有 GPU driver，但 shell 环境没有 CUDA toolkit 路径。

解决：

服务器实际存在：

```text
/usr/local/cuda-11.8/bin/nvcc
/usr/local/cuda-12.4/bin/nvcc
```

当前 torch 是 cu124，因此选择 CUDA 12.4：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4
```

### 6. `mamba-ssm` 预编译 wheel ABI 不匹配

现象：

```text
ImportError: selective_scan_cuda... undefined symbol: _ZN3c107Warning...
```

原因：

安装得到的 `selective_scan_cuda.so` 与当前 `torch==2.6.0+cu124` ABI 不匹配。

尝试从 PyPI 安装 `mamba-ssm==2.3.2.post1` 时，日志出现：

```text
Guessing wheel URL: ... mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE...
```

说明它实际仍在使用预编译 wheel，而不是完全本地编译。

解决方向：

避免继续使用不匹配的 wheel，改用源码编译。

### 7. PyPI sdist 缺少 `selective_scan.cpp`

现象：

```text
ninja: error: ... csrc/selective_scan/selective_scan.cpp, missing and no known rule to make it
```

原因：

从 PyPI sdist 编译 `mamba-ssm==1.2.0.post1` 时，源码包缺少所需 C++ 文件。

解决：

改从 GitHub 仓库源码编译：

```bash
cd /data3/wgq
rm -rf mamba

git clone --recursive https://github.com/state-spaces/mamba.git
cd mamba
git checkout v1.2.0.post1
git submodule update --init --recursive
```

然后在 conda 环境中安装：

```bash
source /data3/wgq/miniconda3/etc/profile.d/conda.sh
conda activate streamvad

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4

python -m pip uninstall -y mamba-ssm causal-conv1d

python -m pip install causal-conv1d==1.2.0.post2 \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

python -m pip install . \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps \
  -v \
  2>&1 | tee /data3/wgq/mamba_build.log
```

### 8. 运行时需要指向 GitHub 源码版 Mamba

验证时实际导入路径为：

```text
/data3/wgq/mamba/mamba_ssm/...
```

因此 smoke 运行时需要加入：

```bash
PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH
```

否则可能继续导入 site-packages 中旧的或不完整的 `mamba_ssm`。

### 9. 其他缺失依赖

逐步补齐的依赖包括：

```bash
python -m pip install \
  einops \
  timm \
  decord \
  lightning \
  torchmetrics \
  Levenshtein \
  accelerate \
  deepspeed \
  sentencepiece \
  protobuf \
  tensorboard \
  imageio \
  imageio-ffmpeg \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`moviepy` 需要使用老版本，否则 `moviepy.editor` 会缺失：

```bash
python -m pip uninstall -y moviepy
python -m pip install moviepy==1.0.3 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 最终验证

Mamba 验证命令：

```bash
PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)

from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
print("mixer_seq_simple OK")

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("selective_scan_fn OK")
PY
```

已通过，输出包括：

```text
torch: 2.6.0+cu124 12.4
mixer_seq_simple OK
selective_scan_fn OK
```

其中 `FutureWarning` 可以暂时忽略，不影响 smoke。

## Stage 1 LoRA Smoke 命令

```bash
cd /data3/wgq/StreamVAD

CUDA_HOME=/usr/local/cuda-12.4 \
PATH=/usr/local/cuda-12.4/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
MAX_SAMPLES=4 \
LOCAL_BATCH_SIZE=1 \
bash tools/run_streamvad_stage1_lora_smoke.sh \
  2>&1 | tee /data3/wgq/streamvad_stage1_lora_smoke.log
```

## 当前结论

当前 StreamMind Stage 1 SSM/Mamba 环境的关键点是：

```text
torch 2.6.0+cu124
CUDA toolkit 12.4
Mamba v1.2.0.post1 GitHub 源码版
PYTHONPATH 指向 /data3/wgq/mamba
transformers==4.37.2
peft==0.10.0
moviepy==1.0.3
```

目前已经确认 `mamba_ssm` 的核心接口可以正常导入，Stage 1 LoRA smoke 的后续问题主要应继续按 `ModuleNotFoundError` 或训练时真实 traceback 逐项处理。
