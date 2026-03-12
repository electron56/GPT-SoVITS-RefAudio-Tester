# GPT-SoVITS RefAudio Tester V2（v1-v4 対応）

[中文说明](README.md)  
[English Version](README_EN.md)

このリポジトリは、GPT-SoVITS の推論段階で参照音声を選別するためのツールです。従来の「リストを読み込む -> バッチごとに試聴する -> 良いサンプルを残す」という WebUI ワークフローを維持しつつ、推論バックエンドを上流の `TTS_infer_pack` に切り替えているため、`v1` から `v4` までの重みに比較的安定して対応できます。

## プロジェクト分析

- 主目的は参照音声の試聴と選別であり、学習フローの統合ではありません。リポジトリには `GPT_SoVITS` 一式が含まれますが、この WebUI 自体は推論時の確認作業に特化しています。
- `webui.py` は、リスト読み込み、ページ送り、話者フィルター、バッチ試聴、モデル切り替え、保存処理を担当します。
- `GPT_SoVITS/inference_main.py` は推論アダプタです。上流 `TTS` パイプラインを包み、読み込まれた SoVITS バージョンに応じて言語候補、`sample_steps`、`super_sampling` を切り替えます。
- 基本フローは `ref.list` を読む -> 現在バッチの試聴音声を生成する -> `採用` を押して元の参照音声をコピーする、です。
- 保存ボタンがコピーするのは右側の生成音声ではなく、左側の元参照音声です。この挙動は誤解されやすいため、README に明記しています。

## 主な機能

- `v1`、`v2`、`v2Pro`、`v2ProPlus`、`v3`、`v4` の重みに対応。
- ページング付きの一括試聴と、実行中に変更できるバッチサイズ（`1-100`）。
- リスト 2 列目の `speaker` を使った話者フィルター。
- 中国語 / 日本語 / 英語の UI 切り替え。
- GPT 重み変更時に、同名の SoVITS 重みを自動で探して切り替え。
- `top_k`、`top_p`、`temperature`、`speed_factor`、`repetition_penalty`、`seed` などの推論パラメータを調整可能。
- `v3` モデルでは `sample_steps` と `super_sampling` を追加表示。

## リポジトリ構成

- `webui.py`：WebUI の入口とリスト管理ロジック。
- `GPT_SoVITS/inference_main.py`：上流 `TTS_infer_pack` の薄いラッパー。
- `GPT_SoVITS/pretrained_models/`：推論時に必要な上流の事前学習済みリソース。
- `SoVITS_weights*` / `GPT_weights*`：バージョン別のモデル重みディレクトリ。
- `ex/`：サンプルのリストファイルと参照音声フォルダ。
- `tools/i18n/`：UI 文言の多言語リソース。

## モデルと事前学習済みリソース

自分のモデル重みを次のいずれかのディレクトリに配置してください。

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

これに加えて、上流 GPT-SoVITS 推論で必要な事前学習済みリソースも必要です。現在の `GPT_SoVITS/configs/tts_infer.yaml` では、少なくとも次の内容を前提にしています。

- `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`
- `GPT_SoVITS/pretrained_models/chinese-hubert-base`
- `gsv-v2final-pretrained`、`gsv-v4-pretrained`、`v2Pro` などのバージョン別事前学習済みディレクトリ

これらが欠けていると、WebUI 自体は起動してもモデル読み込みや推論が失敗します。

## 実行依存

このリポジトリは上流の依存ファイルをそのまま使います。

- `requirements.txt`
- `extra-req.txt`（任意）

まず実行環境に合わせて PyTorch を入れ、その後に残りの依存を入れてください。

```powershell
# NVIDIA GPU（CUDA 12.1）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU のみ
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

補足：

- アプリは `cuda` / `cpu` を自動判定し、CPU 実行時は half precision を自動で無効化します。
- ルートには `ffmpeg` / `ffmpeg.exe` が含まれますが、Python パッケージは別途インストールが必要です。

## 起動

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

例：

```powershell
# リスト内の絶対パスをそのまま使う
python webui.py -l ex/taki.list -b 20

# リストではファイル名だけを使い、実体は指定フォルダから解決する
python webui.py -l ex/taki.list -f ex/ref_mp3 -b 20 -cd -r
```

引数：

- `-l`, `--list`：参照リストのパス。既定値は `ref.list`
- `-p`, `--port`：WebUI のポート。既定値は `14285`
- `-f`, `--folder`：参照音声の基底フォルダ。指定すると、リスト側のディレクトリ部分は捨てられ、ファイル名だけがこのフォルダに結合されます
- `-b`, `--batch`：初期バッチサイズ。範囲は `1-100` で、起動後も UI から変更可能
- `-cd`, `--check_duration`：3 秒未満または 10 秒超の参照音声を除外
- `-r`, `--random_order`：参照リストをシャッフル

補足：

- 起動時にブラウザが自動で開きます。
- Gradio は `0.0.0.0` にバインドされます。LAN 上で実行する場合は、ポート公開範囲を自分で判断してください。

## 参照リスト形式

リストファイルは UTF-8 として読み込まれます。1 行の形式は次のとおりです。

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

各列の意味：

- 1 列目：参照音声のパス。絶対パスでもファイル名だけでも可
- 2 列目：`speaker`。前面 UI のフィルター用であり、モデル側の話者指定ではありません
- 3 列目：参照テキストの言語
- 4 列目：参照テキスト。各行の prompt text として使われます

例：

```text
ref_mp3/sample_001.wav|speaker_a|JA|これはテスト用の参照音声です
```

実際のコード挙動：

- `ZH`、`JP`、`JA`、`EN`、`YUE`、`KO` などの旧表記は、現在の推論パイプラインがサポートする言語ラベルに正規化されます。
- 4 列未満の行は不足分が空文字で補われます。
- 5 列以上ある場合でも使われるのは先頭 4 列だけです。
- `speaker` が空なら `未設定` としてまとめられます。
- `--folder` を使う場合、1 列目は `<folder>/<basename(row[0])>` として解決され、元のサブディレクトリ情報は無視されます。

## 典型的な使い方

1. 重みと事前学習済みリソースを対応ディレクトリへ配置します。
2. `ref.list` を用意するか、`ex/` 配下のサンプルをひな形として使います。
3. `webui.py` を起動し、有効な GPT / SoVITS モデルが読み込まれていることを確認します。
4. 上部のテキストボックスに共通の試聴テキストを入力します。このテキストが現在バッチの各行に対して合成されます。
5. 各行は、自分の参照音声と参照テキストを prompt として使い、その上で試聴テキストの比較音声を生成します。
6. 残したい行で `採用` を押すと、元の参照音声が出力フォルダへコピーされます。既定値は `output/` です。

## WebUI の挙動詳細

- GPT 重みを切り替えると、まず対応する SoVITS ルート内で同じ stem の `.pth` を探します。たとえば `GPT_weights_v3/foo.ckpt` なら `SoVITS_weights_v3/foo.pth` を優先します。
- 合成言語の候補は、現在読み込まれている SoVITS バージョンに応じて変わります。`v2` 以降では `v1` より多く、広東語や韓国語系の選択肢が増えます。
- 開始インデックスのスライダー刻み幅は、現在のバッチサイズに追従します。
- 話者フィルターを変えると、フィルター後リストの先頭からページングし直します。

## 注意事項

- `.pth` または `.ckpt` が 1 つも見つからない場合、アプリはすぐ終了します。
- `--check_duration` は `librosa` を使って 16 kHz で読み込み、3 秒から 10 秒の行だけを残します。
- 試聴テキストが空の場合、生成ボタンを押しても何も出力されません。
- `v3/v4` モデルでは、行ごとの参照テキストが空だとその行が失敗することがあります。
- 1 行の失敗でバッチ全体は止まりません。
- 保存ファイル名は各行の参照テキストから作られ、無効な文字は除去されます。同名になった場合は後から保存したファイルが上書きします。
- リストファイルは UTF-8 固定で開かれます。Shift-JIS や GBK など別エンコーディングのファイルは事前に変換してください。

## License

本リポジトリは [GPLv3](LICENSE) で公開されています。
