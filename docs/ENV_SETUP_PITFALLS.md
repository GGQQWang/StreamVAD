# StreamVAD 环境配置踩坑记录

日期：2026-07-25

## 环境背景

- 服务器：AMAX GPU Linux
- Python：3.10
- CUDA：12.4
- torch：2.6.0+cu124
- streammind conda 环境

## 问题总览

| # | 问题 | 根因 | 关键信号 |
|---|---|---|---|
| 1 | pip 装 flash-attn 后 torch 被覆盖成 2.13.0 | pip 依赖解析器把 torch 升级 | `torch unchanged?` → version jumped |
| 2 | `/tmp` 跨设备 rename 失败 | `/tmp` 和 `/data3` 不同文件系统 | `EXDEV: Invalid cross-device link` |
| 3 | flash-attn wheel ABI 不匹配 | `cxx11abiFALSE` vs `cxx11abiTRUE` | `undefined symbol: _ZN3c105Error...` |
| 4 | 源码编译 ABI 不对 | CXXFLAGS 没传对或 torch 本身 ABI 已残 | 同上 |
| 5 | torch libc10.so 被反复覆盖半残 | torch 2.13 ↔ 2.6 来回装 | libc10 符号是旧 ABI，属性返回新 ABI |
| 6 | requirements.txt 版本过时 | 固定 torch==2.2.0，与 CUDA 12.4 不兼容 | — |
| 7 | transformers 版本冲突 | >= 4.44.2 缺 TRANSFORMERS_CACHE | `cannot import name 'TRANSFORMERS_CACHE'` |
| 8 | peft 版本冲突 | 新 peft 需要新 transformers API | `cannot import name 'EncoderDecoderCache'` |
| 9 | mamba-ssm PyPI wheel ABI 不匹配 | 预编译 wheel 与 torch ABI 不一致 | `undefined symbol: _ZN3c107Warning...` |
| 10 | scikit-learn-intelex 版本不存在 | 图中版本号写死了一个不存在的 release | `No matching distribution found` |
| 11 | 裸 pip 装到系统环境 | `which pip` 指向系统环境而非 conda | 包装到了 `/usr/lib/python3` |
| 12 | PyPI 超时 | 国外源网络慢 | `ReadTimeoutError` |

---

## 逐问题详解

### 1. pip 装 flash-attn 时 torch 被升级

**现象：**

```
torchvision 0.21.0+cu124 requires torch==2.6.0, but you have torch 2.13.0
```

**根因：**

`flash-attn==2.5.8` 的 `setup.py` 里写了 `install_requires=['torch']`，没有版本上限。pip 的依赖解析器看到 torch 2.6.0 和它认识的版本"不匹配"，就自动升级到最新版（2.13.0）。

**解决方案：**

用预编译 wheel 时加 `--no-deps`：

```bash
python -m pip install --no-deps <wheel-url>
```

源码编译时不会触发这个问题（因为编译过程中不解析 runtime 依赖），但装预编译 wheel 必须带 `--no-deps`。

**关键教训：**

安装任何带 C++ extension 的包时，先 `pip install --dry-run <pkg>` 看它会拉什么依赖。用 `--no-deps` 或 `constraints.txt` 锁死 torch 版本。

---

### 2. /tmp 跨设备 rename（EXDEV）

**现象：**

```
error: [Errno 18] Invalid cross-device link: 'xxx.whl' -> '/home/wgq/.cache/pip/wheels/.../xxx.whl'
```

**根因：**

pip 先把 wheel 下载到 `/tmp`，然后用 `os.rename()` 移到 `/home` 或 `/data3` 下的缓存目录。Linux `rename(2)` 系统调用不支持跨文件系统操作，返回 `EXDEV`（errno 18）。

**解决方案：**

方案 1（最简单）：跳过缓存

```bash
pip install xxx --no-cache-dir
```

方案 2：把临时目录设到同一文件系统

```bash
export TMPDIR=/data3/wgq/tmp && mkdir -p $TMPDIR
pip install xxx
```

方案 3：pip 永久配置

```ini
# ~/.pip/pip.conf
[global]
build = /data3/wgq/tmp
```

---

### 3. flash-attn 预编译 wheel ABI 不匹配

**现象：**

```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

**根因：**

flash-attn 的 GitHub Release 提供两种预编译 wheel：

- `cxx11abiTRUE` — 新 ABI（匹配官方 PyTorch）
- `cxx11abiFALSE` — 旧 ABI

如果下载了不匹配的版本，C++ 符号对不上，动态加载失败。

**诊断方法：**

```bash
# 查看 torch 的 ABI
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"   # True = 新, False = 旧

# 查看 libc10.so 导出的符号（确认 ABI）
nm -D $(python -c "import torch; print(torch.__file__.replace('__init__.py','lib/libc10.so'))") | grep "c10.*Error"
```

`Ss` 结尾 = 旧 ABI，`NSt7__cxx11...` = 新 ABI。

装预编译 wheel 时必须选对：

```bash
# 官方 PyTorch → ABI=True
python -m pip install --no-deps <url-with-cxx11abiTRUE>

# 自编译 PyTorch → ABI=False
python -m pip install --no-deps <url-with-cxx11abiFALSE>
```

源码编译则无需关心 — 会自动匹配当前 torch 的 ABI。

---

### 4. torch libc10.so 被反复覆盖残废

**现象：**

- `torch._C._GLIBCXX_USE_CXX11_ABI` 返回 `True`
- 但 `libc10.so` 导出的符号是旧 ABI（`Ss`）

**根因：**

torch 2.13 覆盖 2.6 之后又卸载重装，旧 `.so` 文件可能没完全清理。

**解决方案：**

这种情况下直接开新 conda 环境，不要在污染的环境里修复。用 `conda create -n streamvad_fresh` 重来。

---

### 5. requirements.txt 版本与 CUDA 12.4 不兼容

**现象：**

requirements.txt 固定 `torch==2.2.0`、`torchvision==0.17.0`，不支持 CUDA 12.4。

**根因：**

StreamMind 的 `requirements.txt` 是早期版本，写死了老 torch。但 README 说 `>=2.5.1`。两个互相矛盾。

**解决方案：**

torch 不写进 requirements.txt，单独从 PyTorch 官方索引安装：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

然后用 `grep -v '^torch\|^torchvision' requirements.txt | xargs pip install` 安装其余包，跳过 torch。

---

### 6. transformers 和 peft 版本连锁冲突

**现象：**

```text
cannot import name 'TRANSFORMERS_CACHE' from 'transformers'
cannot import name 'EncoderDecoderCache' from 'transformers'
```

**根因：**

- StreamMind 旧代码用 `from transformers import TRANSFORMERS_CACHE`，新版 transformers 移除了此导出 → 需要 `transformers==4.37.2`
- 但较新的 peft 依赖较新的 transformers API → 需要 `peft==0.10.0`

**解决方案：**

```bash
python -m pip install transformers==4.37.2 peft==0.10.0
```

---

### 7. mamba-ssm PyPI 预编译 wheel ABI 不匹配

**现象：**

```
ImportError: selective_scan_cuda... undefined symbol: _ZN3c107Warning...
```

**根因：**

PyPI 上的 mamba-ssm wheel 编译时的 torch ABI 与当前环境不一致。

**解决方案：**

从 GitHub 源码编译 v1.2.0.post1：

```bash
git clone --recursive https://github.com/state-spaces/mamba.git /data3/wgq/mamba
cd /data3/wgq/mamba && git checkout v1.2.0.post1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MAX_JOBS=4
python -m pip install . --no-build-isolation --no-cache-dir --no-deps
```

使用时必须带 `PYTHONPATH`：

```bash
PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH python ...
```

---

### 8. 裸 pip 装到系统环境

**现象：**

conda activate 之后 pip 装包，包出现在系统目录而非 conda 环境。

**根因：**

`which pip` 指向 `/bin/pip` 或 `/usr/bin/pip`，不是 conda 环境里的 pip。

**解决方案：**

始终使用 `python -m pip install ...`，确保用的是当前 Python 解释器对应的 pip。

---

### 9. PyPI 超时

**现象：**

```
ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

**解决方案：**

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <packages>
```

---

## 正确安装顺序（新环境）

```bash
# 1. 新建 conda 环境
conda create -n streamvad_fresh python=3.10 -y
conda activate streamvad_fresh

# 2. PyTorch（先装，独立于 requirements）
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. 验证 ABI
python -c "import torch; print('ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
# 必须返回 True

# 4. 基础依赖（跳过 torch 相关）
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  accelerate==0.26.1 decorator==4.4.2 decord==0.6.0 deepspeed==0.13.1 \
  distributed==2022.7.0 numpy==1.24.4 scikit-image==0.19.3 scikit-learn==1.2.2 \
  scikit-video==1.1.11 scipy==1.10.1 Scrapy==2.8.0 seaborn==0.12.2 \
  torchmetrics==1.4.3 tornado==6.1 tqdm==4.64.1 traitlets==5.7.1 \
  pytorch-lightning==2.4.0 lightning==2.2.2 scenedetect==0.6.3 \
  einops timm Levenshtein sentencepiece protobuf tensorboard imageio imageio-ffmpeg

# 5. transformers 和 peft（锁定兼容版本）
python -m pip install transformers==4.37.2 peft==0.10.0 moviepy==1.0.3

# 6. mamba-ssm（GitHub 源码编译）
git clone --recursive https://github.com/state-spaces/mamba.git /data3/wgq/mamba
cd /data3/wgq/mamba && git checkout v1.2.0.post1
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$CUDA_HOME/bin:$PATH \
  LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH MAX_JOBS=4
python -m pip install . --no-build-isolation --no-cache-dir --no-deps

# 7. flash-attn（源码编译，ABI 自动匹配）
export TMPDIR=/data3/wgq/tmp && mkdir -p $TMPDIR
python -m pip install flash-attn==2.5.8 --no-build-isolation --no-cache-dir

# 8. 最终验证
PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH python -c "
import torch; print('torch:', torch.__version__, torch.version.cuda)
import flash_attn; print('flash_attn OK')
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights; print('mamba OK')
print('ALL OK')
"
```

## 运行时环境变量

每次跑 StreamVAD 训练/推理需要提前设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/data3/wgq/mamba:$PYTHONPATH
```

可写入 `~/.bashrc` 或单独的 `env.sh` source 文件。

## 关键习惯总结

| 习惯 | 原因 |
|---|---|
| 用 `python -m pip` 而非裸 `pip` | 避免装到系统环境 |
| 预编译 wheel 加 `--no-deps` | 防止 pip 覆盖 torch |
| 装前 `--dry-run` 看依赖 | 提前发现版本冲突 |
| 装后立即 `python -c "import x; print('OK')"` | 第一时间发现问题 |
| 源码编译包用 `TMPDIR` | 避免跨设备 rename |
| 遇到 ABI 报错先看 `_GLIBCXX_USE_CXX11_ABI` | 确定 ABI 方向 |
| 环境被污染直接 `conda create -n xxx_fresh` | 修不如换 |
| 国外源超时加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` | 提速 |
| 版本号不写死的 pip 包（如 `scikit-learn-intelex==20230228.214242`） | 可能根本不存在 |

## 关键诊断命令

```bash
# ABI 检查
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"

# libc10 符号检查
nm -D $(python -c "import torch; print(torch.__file__.replace('__init__.py','lib/libc10.so'))") | grep "c10.*Error"

# flash-attn 符号检查
python -c "import flash_attn" 2>&1 | grep "undefined symbol"

# 查看 pip 会拉什么依赖
python -m pip install --dry-run <package>

# 确认包装到哪了
python -m pip show <package>
```
