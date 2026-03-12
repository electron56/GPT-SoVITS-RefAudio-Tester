# GPT-SoVITS RefAudio Tester V2 (v1-v4 Compatible)

[中文说明](README.md)  
[日本語版](README_JA.md)

This repository is a reference-audio selection tool for the GPT-SoVITS inference stage. It keeps the original WebUI workflow of "load a list -> preview in batches -> keep accepted samples", while switching the backend to the upstream `TTS_infer_pack`, so it can work more reliably across `v1` to `v4` model families.

## Project analysis

- The focus is reference-audio testing and triage, not training orchestration. The repo includes the full `GPT_SoVITS` tree, but this WebUI is mainly for inference-time preview.
- `webui.py` handles list loading, paging, speaker filtering, batch preview, model switching, and save actions.
- `GPT_SoVITS/inference_main.py` is the inference adapter. It wraps the upstream `TTS` pipeline and adjusts language choices, `sample_steps`, and `super_sampling` based on the loaded SoVITS version.
- The core workflow is: read `ref.list` -> generate preview audio for the current batch -> click `Accept` to copy the original reference audio.
- The save button copies the original reference audio, not the generated audition audio. That behavior is easy to misunderstand, so it is documented explicitly here.

## Main capabilities

- Supports `v1`, `v2`, `v2Pro`, `v2ProPlus`, `v3`, and `v4` weights.
- Batch preview with pageable lists and runtime-adjustable batch size (`1-100`).
- Speaker-based filtering using the `speaker` column from the list file.
- UI language switching for Chinese / Japanese / English.
- Automatic same-name SoVITS matching when the GPT model is changed.
- Exposes inference controls such as `top_k`, `top_p`, `temperature`, `speed_factor`, `repetition_penalty`, and `seed`.
- Shows `sample_steps` and `super_sampling` for `v3` models.

## Repository layout

- `webui.py`: WebUI entry point and list-management logic.
- `GPT_SoVITS/inference_main.py`: thin wrapper around upstream `TTS_infer_pack`.
- `GPT_SoVITS/pretrained_models/`: upstream pretrained assets required at inference time.
- `SoVITS_weights*` / `GPT_weights*`: version-grouped model weight directories.
- `ex/`: example list files and sample reference-audio folder.
- `tools/i18n/`: localized UI text resources.

## Models and pretrained assets

Put your own model weights into any of the following directories:

- SoVITS (`.pth`)
  - `SoVITS_weights`
  - `SoVITS_weights_v2`
  - `SoVITS_weights_v2Pro`
  - `SoVITS_weights_v2ProPlus`
  - `SoVITS_weights_v3`
  - `SoVITS_weights_v4`
- GPT (`.ckpt`)
  - `GPT_weights`
  - `GPT_weights_v2`
  - `GPT_weights_v2Pro`
  - `GPT_weights_v2ProPlus`
  - `GPT_weights_v3`
  - `GPT_weights_v4`

You also need the upstream GPT-SoVITS pretrained assets. The current `GPT_SoVITS/configs/tts_infer.yaml` expects at least:

- `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`
- `GPT_SoVITS/pretrained_models/chinese-hubert-base`
- version-specific pretrained directories such as `gsv-v2final-pretrained`, `gsv-v4-pretrained`, and `v2Pro`

If those assets are missing, the WebUI may launch but model loading or inference will fail.

## Runtime dependencies

This repository uses the upstream dependency files:

- `requirements.txt`
- `extra-req.txt` (optional)

Install PyTorch for your target runtime first, then install the remaining packages:

```powershell
# NVIDIA GPU (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

Notes:

- The app auto-detects `cuda` vs `cpu`; half precision is disabled automatically on CPU.
- `ffmpeg` / `ffmpeg.exe` is bundled in the repository root, but Python packages still need to be installed separately.

## Run

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

Examples:

```powershell
# Use absolute paths from the list file directly
python webui.py -l ex/taki.list -b 20

# Use only basenames from the list and resolve them from a folder
python webui.py -l ex/taki.list -f ex/ref_mp3 -b 20 -cd -r
```

Arguments:

- `-l`, `--list`: reference list path, default `ref.list`
- `-p`, `--port`: WebUI port, default `14285`
- `-f`, `--folder`: base folder for reference audio. When set, the app discards directory parts from the list path and joins only the basename to this folder
- `-b`, `--batch`: initial batch size, range `1-100`, also adjustable in the UI
- `-cd`, `--check_duration`: keep only reference audios between 3 and 10 seconds
- `-r`, `--random_order`: shuffle the reference list

Extra behavior:

- The app opens a browser window automatically on launch.
- Gradio binds to `0.0.0.0`. If you run it on a LAN-accessible machine, evaluate the port exposure yourself.

## Reference list format

The list file is read as UTF-8. Each row is:

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

Fields:

- column 1: reference-audio path, either an absolute path or just a filename
- column 2: `speaker`, used only for frontend filtering, not for model-side speaker selection
- column 3: reference text language
- column 4: reference text, which becomes the prompt text for that row

Example:

```text
ref_mp3/sample_001.wav|speaker_a|JA|This is a sample reference line
```

Actual code behavior:

- Legacy language tags such as `ZH`, `JP`, `JA`, `EN`, `YUE`, and `KO` are normalized to the labels supported by the current inference pipeline.
- Rows with fewer than 4 columns are padded with empty strings.
- Rows with more than 4 columns only use the first 4 fields.
- Empty `speaker` values are grouped as `Unlabeled`.
- With `--folder`, column 1 is resolved as `<folder>/<basename(row[0])>`, so original subdirectories from the list are ignored.

## Typical workflow

1. Put weights and pretrained assets into their expected directories.
2. Prepare `ref.list`, or start from one of the templates under `ex/`.
3. Launch `webui.py` and confirm that valid GPT and SoVITS models are loaded.
4. Enter a shared audition text in the top text box. That text will be synthesized against every row in the current batch.
5. Each row uses its own reference audio and reference text as the prompt, then generates a comparison sample from the audition text.
6. Click `Accept` for rows you want to keep. The app copies the original reference audio into the output folder, default `output/`.

## WebUI behavior details

- When GPT weights change, the app first tries to find a same-stem `.pth` in the paired SoVITS root. For example, `GPT_weights_v3/foo.ckpt` prefers `SoVITS_weights_v3/foo.pth`.
- The synthesis-language dropdown changes with the loaded SoVITS version. `v2+` exposes more choices than `v1`, including Cantonese and Korean related options.
- The start-index slider step follows the current batch size.
- Switching the speaker filter resets paging to the start of the filtered subset.

## Notes

- If no `.pth` or `.ckpt` files are found, the app exits immediately.
- `--check_duration` uses `librosa` at 16 kHz and keeps only entries between 3 and 10 seconds.
- If the audition text is empty, clicking generate returns no outputs.
- For `v3/v4` models, a row with empty prompt text may fail.
- A failed row does not stop the rest of the batch.
- Saved filenames come from that row's reference text with invalid filesystem characters removed. Duplicate names will overwrite older files.
- The list file is opened as UTF-8 only. Convert Shift-JIS, GBK, or other encodings before use if needed.

## License

This repository is released under [GPLv3](LICENSE).
