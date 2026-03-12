# GPT-SoVITS RefAudio Tester V2（兼容 v1-v4）

[English Version](README_EN.md)  
[日本語版](README_JA.md)

本仓库是一个面向 GPT-SoVITS 推理阶段的参考音频筛选工具。它保留了“批量加载参考列表 -> 逐批试听 -> 一键保留满意样本”的 WebUI 工作流，同时将底层推理切换到上游 `TTS_infer_pack`，因此可以更稳定地兼容 `v1` 到 `v4` 系列权重。

## 项目分析

- 项目定位是“参考音频测试与挑选”，不是训练脚本封装器。仓库里虽然带有完整 `GPT_SoVITS` 目录，但这个 WebUI 主要服务于推理阶段的批量试听。
- `webui.py` 负责列表读取、分页、说话人筛选、批量试听、模型切换和保存操作。
- `GPT_SoVITS/inference_main.py` 是当前推理适配层，统一调用上游 `TTS` 管线，并根据已加载 SoVITS 版本动态调整可用语言、`sample_steps` 和 `super_sampling`。
- 核心工作流是：读取 `ref.list` -> 为当前批次生成试听音频 -> 点击“满意/Accept/採用”复制原始参考音频。
- 保存按钮复制的是原始参考音频，不是右侧生成出来的试听音频。这一点和很多“导出合成结果”的工具不同，README 里单独说明比较重要。

## 主要能力

- 兼容 `v1`、`v2`、`v2Pro`、`v2ProPlus`、`v3`、`v4` 权重。
- 支持批量分页试听，前端可动态调整每批数量（`1-100`）。
- 支持按 `speaker` 字段筛选参考音频。
- 支持中文 / 日本語 / English 三种界面语言。
- 支持在切换 GPT 权重时自动寻找同名 SoVITS 权重并联动切换。
- 支持 `top_k`、`top_p`、`temperature`、`speed_factor`、`repetition_penalty`、`seed` 等推理参数。
- `v3` 模型额外显示 `sample_steps` 和 `super_sampling`。

## 仓库结构

- `webui.py`：WebUI 入口，负责参数解析、列表管理和页面交互。
- `GPT_SoVITS/inference_main.py`：对上游 `TTS_infer_pack` 的轻量封装。
- `GPT_SoVITS/pretrained_models/`：上游推理依赖的预训练资源目录。
- `SoVITS_weights*` / `GPT_weights*`：按版本分类的模型权重目录。
- `ex/`：示例列表文件和示例参考音频目录。
- `tools/i18n/`：界面文案的多语言资源。

## 模型与预训练资源

请把你自己的模型权重放入以下任意目录：

- SoVITS（`.pth`）
  - `SoVITS_weights`
  - `SoVITS_weights_v2`
  - `SoVITS_weights_v2Pro`
  - `SoVITS_weights_v2ProPlus`
  - `SoVITS_weights_v3`
  - `SoVITS_weights_v4`
- GPT（`.ckpt`）
  - `GPT_weights`
  - `GPT_weights_v2`
  - `GPT_weights_v2Pro`
  - `GPT_weights_v2ProPlus`
  - `GPT_weights_v3`
  - `GPT_weights_v4`

除此之外，还需要准备上游 `GPT-SoVITS` 推理所需的预训练资源。当前配置文件 `GPT_SoVITS/configs/tts_infer.yaml` 默认会读取至少以下内容：

- `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`
- `GPT_SoVITS/pretrained_models/chinese-hubert-base`
- 各版本默认预训练权重目录，例如 `gsv-v2final-pretrained`、`gsv-v4-pretrained`、`v2Pro`

如果这些资源缺失，WebUI 即使能启动，模型加载或推理也会失败。

## 运行依赖

本仓库沿用上游依赖文件：

- `requirements.txt`
- `extra-req.txt`（可选）

建议先按你的运行环境安装 PyTorch，再安装其余依赖：

```powershell
# NVIDIA GPU（CUDA 12.1）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

说明：

- 程序会自动检测 `cuda`/`cpu`；若实际运行在 CPU 上，会自动关闭 half precision。
- 根目录已附带 `ffmpeg`/`ffmpeg.exe`，但 Python 依赖仍需要按上面的方式安装。

## 启动方式

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

示例：

```powershell
# 直接使用列表中的绝对路径
python webui.py -l ex/taki.list -b 20

# 只使用列表中的文件名，并从指定目录查找音频
python webui.py -l ex/taki.list -f ex/ref_mp3 -b 20 -cd -r
```

参数说明：

- `-l`, `--list`：参考列表文件路径，默认 `ref.list`
- `-p`, `--port`：WebUI 端口，默认 `14285`
- `-f`, `--folder`：参考音频基目录。启用后，程序会忽略列表里原路径的目录部分，只取文件名再拼接到该目录
- `-b`, `--batch`：初始每批数量，范围 `1-100`，启动后也可在前端调整
- `-cd`, `--check_duration`：过滤掉不在 3-10 秒范围内的参考音频
- `-r`, `--random_order`：打乱参考列表顺序

补充说明：

- WebUI 启动时会自动打开浏览器。
- Gradio 监听地址是 `0.0.0.0`。如果你在局域网环境运行，请自行评估暴露端口的风险。

## 参考列表格式

列表文件按 UTF-8 读取，每行格式为：

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

字段说明：

- 第 1 列：参考音频路径，既可以是绝对路径，也可以只是文件名
- 第 2 列：`speaker`，仅用于前端筛选，不参与模型推理
- 第 3 列：参考文本语言
- 第 4 列：参考文本，也就是该参考音频对应的 prompt text

示例：

```text
ref_mp3/sample_001.wav|speaker_a|JA|これはテスト用の参照音声です
```

代码中的实际行为：

- 旧语言标记如 `ZH`、`JP`、`JA`、`EN`、`YUE`、`KO` 会自动规范化为当前推理管线支持的语言标签。
- 如果某行列数少于 4，缺失列会补空字符串。
- 如果某行列数超过 4，程序只使用前 4 列。
- 如果 `speaker` 为空，会被归类为“未标注”。
- 如果使用 `--folder`，程序会把第 1 列解析成 `<folder>/<basename(row[0])>`，原有子目录信息会被丢弃。

## 使用流程

1. 将权重和预训练资源放到对应目录。
2. 准备 `ref.list`，或直接使用 `ex/*.list` 作为模板。
3. 启动 `webui.py`，确认左上角已加载到可用的 GPT / SoVITS 模型。
4. 在“试听文本”里输入一段统一的测试文本，这段文本会对当前批次中的所有参考音频逐条合成。
5. 每一行会使用自己的参考音频和参考文本作为 prompt，再用上面的“试听文本”生成对比音频。
6. 听完后点击“满意”，程序会把左侧原始参考音频复制到输出目录，默认是 `output/`。

## WebUI 行为细节

- 切换 GPT 权重时，程序会优先在配对目录中寻找同名 `.pth`，例如 `GPT_weights_v3/foo.ckpt` 会优先尝试匹配 `SoVITS_weights_v3/foo.pth`。
- 可选的“合成语言”会随着当前 SoVITS 版本动态变化。`v2` 及以上会比 `v1` 多出粤语、韩语等选项。
- 起始索引滑块的步长会随当前每批数量变化。
- 说话人筛选切换后，会从筛选结果的第 0 条重新开始分页。

## 注意事项

- 如果没有找到任何 `.pth` 或 `.ckpt`，程序会直接退出。
- `--check_duration` 使用 `librosa` 以 16k 采样率读取音频，并只保留 3-10 秒的条目。
- 如果“试听文本”为空，点击生成不会产生任何输出。
- 对于 `v3/v4` 模型，如果某一行的参考文本为空，该行可能失败。
- 某一行推理失败不会中断整批，其它行仍会继续生成。
- 保存文件时，文件名来自该行的参考文本，非法字符会被移除；如果多个条目得到相同文件名，后保存的文件会覆盖前一个。
- 代码当前将列表文件按 UTF-8 打开。如果你的列表是 Shift-JIS、GBK 等编码，需要先自行转换。

## License

本仓库使用 [GPLv3](LICENSE)。
