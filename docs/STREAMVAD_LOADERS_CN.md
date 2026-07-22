# StreamVAD Loader 说明

当前新增了两个 PyTorch-style loader，位置：

- `streamvad/data/datasets.py`

它们默认只返回训练需要的白名单字段，避免把审计字段、路径标签、原始边界等信息喂给模型。

## Stage 1 Loader

类名：

```python
StreamVADStage1Dataset
```

默认返回：

```python
{
    "video": "...",
    "clip_start": 0.0,
    "clip_end": 10.0,
    "target_text": "Scene prior:\n...\n\nObservation:\n...\n\nAnswer:\nNormal"
}
```

训练时只应该使用：

- `video`
- `clip_start`
- `clip_end`
- `target_text`

## Stage 2 Gate Loader

类名：

```python
StreamVADStage2GateDataset
```

默认返回：

```python
{
    "video": "...",
    "chunk_start": 4.0,
    "chunk_end": 5.0,
    "gate_label": 1
}
```

训练时只应该使用：

- `video`
- `chunk_start`
- `chunk_end`
- `gate_label`

`gate_label` 取值：

- `0 = silence`
- `1 = response`
- `-100 = ignore`

Gate loss 必须使用：

```python
torch.nn.CrossEntropyLoss(weight=..., ignore_index=-100)
```

## 调试用法

不解码视频，只检查 JSONL 字段：

```python
from streamvad.data import StreamVADStage1Dataset, StreamVADStage2GateDataset

s1 = StreamVADStage1Dataset(
    "data/streamvad_weak_supervision/streamvad_stage1_train.jsonl",
    require_video_exists=False,
)

s2 = StreamVADStage2GateDataset(
    "data/streamvad_weak_supervision/streamvad_stage2_train.jsonl",
    require_video_exists=False,
)

print(s1[0])
print(s2[0])
```

正式训练前建议打开：

```python
require_video_exists=True
require_reliable_timing=True
```

需要直接解码视频片段时，再设置：

```python
decode_video=True
```

这会懒加载 `torch` 和 `torchvision`，并按 `clip_start/clip_end` 或 `chunk_start/chunk_end` 读取对应视频片段。

## 不应进入模型的字段

以下字段只用于审计、统计或追溯，不应拼进 prompt，也不应作为模型输入：

- `original_think`
- `original_answer`
- `original_start`
- `original_end`
- `original_video`
- `video_id`
- `video_key`
- `trigger_reason`
- `trigger_sources`
- `gate_text`
- `original_source`
