# My Understanding
## 审计
- Stage2只训练projector中名称含cls的参数。风险是现在是字符串匹配，后期可改为model.mm_projector.cls_net.parameters()；
- Gate 是浅层 Mistral，不是普通线性二分类头；
- SoccerNet的silence/response标签不是数据集直接提供的，而gate学是当前视觉信息是否已经到达应该生成新 caption 的时刻；
- 大问题：VAD-R1 的完整 CoT 用于在线前缀会泄漏未来信息；
- 环境冲突，两个仓库的环境不能简单合并；
- SoccerNet Stage 2 实际走 fallback 分支，Gate 使用 0/1 标签；32000/32001 只存在于未被当前 SoccerNet 路径触发的 prompt-template 分支。该分支若直接执行，可能与二分类 logits 冲突，属于后续改造时需要规避的潜在问题；
- `主程序里的 model
      │
      │ model.stream_generate_demo(...)
      ▼
self = model
      │
      │ self.frame_feature 是历史缓存
      ▼
prepare_inputs...
      │
      │ 调用 encode_images...
      ▼
当前帧经过 CLIP
      │
      │ 与历史 frame_feature 拼接
      ▼
temporal_aggregator(frames_features, cls_demo=True)
      │
      │ 转交 mm_projector
      ▼
Video_Mamba_seq
      ├── Mamba 生成视觉特征
      └── ClsNet 生成 cls_feature `
- 