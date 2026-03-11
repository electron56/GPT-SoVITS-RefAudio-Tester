import argparse
import csv
import logging
import os
import random
import re
import shutil
from typing import Dict, List, Optional

import gradio as gr
import librosa
from tqdm import tqdm

from GPT_SoVITS import inference_main

logging.getLogger("PIL.Image").propagate = False

SOVITS_WEIGHT_ROOTS = [
    "SoVITS_weights",
    "SoVITS_weights_v2",
    "SoVITS_weights_v3",
    "SoVITS_weights_v4",
    "SoVITS_weights_v2Pro",
    "SoVITS_weights_v2ProPlus",
]

GPT_WEIGHT_ROOTS = [
    "GPT_weights",
    "GPT_weights_v2",
    "GPT_weights_v3",
    "GPT_weights_v4",
    "GPT_weights_v2Pro",
    "GPT_weights_v2ProPlus",
]

CUT_METHOD_OPTIONS = [
    ("no_cut", 0),
    ("every_4_sentences", 1),
    ("every_50_chars", 2),
    ("zh_full_stop", 3),
    ("en_full_stop", 4),
    ("punctuation", 5),
]
CUT_METHOD_VALUE_BY_ID = {item_id: value for item_id, value in CUT_METHOD_OPTIONS}

LEGACY_REF_LANGUAGE_ALIASES = {
    "ZH": "Chinese",
    "zh": "Chinese",
    "JP": "Japanese",
    "jp": "Japanese",
    "JA": "Japanese",
    "ja": "Japanese",
    "EN": "English",
    "en": "English",
    "En": "English",
    "YUE": "Cantonese",
    "yue": "Cantonese",
    "KO": "Korean",
    "ko": "Korean",
}

# Runtime state
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
g_ref_folder = ""
g_batch = 10
g_index = 0
g_ref_list: List[Dict[str, str]] = []
g_ref_list_max_index = 0
g_ref_audio_path_list: List[Optional[str]] = []
g_SoVITS_names: List[str] = []
g_GPT_names: List[str] = []
g_ui_language = "zh"

UI_TEXTS = {
    "zh": {
        "title": "# GPT-SoVITS 参考音频测试 WebUI",
        "subtitle": "批量参考音频预览，兼容 GPT-SoVITS v1-v4。",
        "ui_language": "界面语言",
        "model_selection": "## 模型选择",
        "gpt_model": "GPT 模型",
        "sovits_model": "SoVITS 模型",
        "output_dir": "满意的参考音频复制到",
        "refresh_models": "刷新模型",
        "model_version": "模型版本",
        "status": "状态",
        "ready": "就绪",
        "synthesis_options": "## 合成选项",
        "preview_text": "试听文本",
        "preview_placeholder": "用以合成试听音频的文本",
        "synthesis_language": "合成语言",
        "split_method": "切分方式",
        "sample_steps_v3": "sample_steps（仅 v3）",
        "super_sampling_v3": "super_sampling（仅 v3）",
        "batch_preview": "## 试听批次",
        "start_index": "起始索引",
        "batch_size": "每批数量",
        "prev_batch": "上一批",
        "next_batch": "下一批",
        "generate_preview": "生成试听语音",
        "preview_list": "## 试听列表",
        "ref_audio": "参考音频",
        "ref_language": "参考文本语言",
        "ref_text": "参考文本",
        "generated": "试听音频",
        "save": "满意",
        "saved": "已保存",
        "save_failed": "保存失败",
        "version_text": "SoVITS 版本：{version}",
        "status_refreshed": "已刷新。SoVITS：{sovits} | GPT：{gpt}",
        "status_loaded_sovits": "已加载 SoVITS：{name}",
        "status_failed_sovits": "加载 SoVITS 失败：{error}",
        "status_loaded_gpt": "已加载 GPT：{name}",
        "status_failed_gpt": "加载 GPT 失败：{error}",
        "status_auto_switched_sovits": "已加载 GPT：{gpt}，并自动切换 SoVITS：{sovits}",
        "status_auto_switch_sovits_failed": "已加载 GPT：{gpt}，自动切换 SoVITS 失败：{error}",
        "status_no_paired_sovits": "已加载 GPT：{gpt}，未找到同名 SoVITS。",
        "no_model_found": "未找到模型。请将 .pth 放入 SoVITS_weights*，并将 .ckpt 放入 GPT_weights*。",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_en": "English",
        "cut_no_cut": "不切分",
        "cut_every_4_sentences": "每 4 句切分",
        "cut_every_50_chars": "每 50 字切分",
        "cut_zh_full_stop": "按中文句号切分",
        "cut_en_full_stop": "按英文句号切分",
        "cut_punctuation": "按标点切分",
    },
    "ja": {
        "title": "# GPT-SoVITS 参照音声テスター WebUI",
        "subtitle": "参照音声の一括プレビュー。GPT-SoVITS v1-v4 に対応。",
        "ui_language": "UI 言語",
        "model_selection": "## モデル選択",
        "gpt_model": "GPT モデル",
        "sovits_model": "SoVITS モデル",
        "output_dir": "採用した参照音声のコピー先",
        "refresh_models": "モデルを更新",
        "model_version": "モデルバージョン",
        "status": "ステータス",
        "ready": "準備完了",
        "synthesis_options": "## 合成オプション",
        "preview_text": "試聴テキスト",
        "preview_placeholder": "試聴音声の合成に使うテキストを入力",
        "synthesis_language": "合成言語",
        "split_method": "テキスト分割方法",
        "sample_steps_v3": "sample_steps（v3 のみ）",
        "super_sampling_v3": "super_sampling（v3 のみ）",
        "batch_preview": "## 試聴バッチ",
        "start_index": "開始インデックス",
        "batch_size": "バッチ件数",
        "prev_batch": "前のバッチ",
        "next_batch": "次のバッチ",
        "generate_preview": "試聴音声を生成",
        "preview_list": "## 試聴一覧",
        "ref_audio": "参照音声",
        "ref_language": "参照テキスト言語",
        "ref_text": "参照テキスト",
        "generated": "試聴音声",
        "save": "採用",
        "saved": "保存済み",
        "save_failed": "保存失敗",
        "version_text": "SoVITS バージョン: {version}",
        "status_refreshed": "更新完了。SoVITS: {sovits} | GPT: {gpt}",
        "status_loaded_sovits": "SoVITS を読み込みました: {name}",
        "status_failed_sovits": "SoVITS の読み込みに失敗: {error}",
        "status_loaded_gpt": "GPT を読み込みました: {name}",
        "status_failed_gpt": "GPT の読み込みに失敗: {error}",
        "status_auto_switched_sovits": "GPT を読み込み、同名 SoVITS へ自動切替: GPT={gpt}, SoVITS={sovits}",
        "status_auto_switch_sovits_failed": "GPT は読み込み済みですが、SoVITS 自動切替に失敗: {error}",
        "status_no_paired_sovits": "GPT を読み込みましたが、同名 SoVITS が見つかりません: {gpt}",
        "no_model_found": "モデルが見つかりません。.pth を SoVITS_weights* に、.ckpt を GPT_weights* に配置してください。",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_en": "English",
        "cut_no_cut": "分割なし",
        "cut_every_4_sentences": "4 文ごとに分割",
        "cut_every_50_chars": "50 文字ごとに分割",
        "cut_zh_full_stop": "中国語句点で分割",
        "cut_en_full_stop": "英語ピリオドで分割",
        "cut_punctuation": "句読点で分割",
    },
    "en": {
        "title": "# GPT-SoVITS RefAudio Tester WebUI",
        "subtitle": "Batch reference-audio preview with compatibility for GPT-SoVITS v1-v4.",
        "ui_language": "UI Language",
        "model_selection": "## Model Selection",
        "gpt_model": "GPT Model",
        "sovits_model": "SoVITS Model",
        "output_dir": "Copy accepted reference audio to",
        "refresh_models": "Refresh Models",
        "model_version": "Model Version",
        "status": "Status",
        "ready": "Ready",
        "synthesis_options": "## Audition Options",
        "preview_text": "Audition Text",
        "preview_placeholder": "Text used to synthesize audition audio",
        "synthesis_language": "Synthesis Language",
        "split_method": "Text Split Method",
        "sample_steps_v3": "sample_steps (v3 only)",
        "super_sampling_v3": "super_sampling (v3 only)",
        "batch_preview": "## Audition Batch",
        "start_index": "Start Index",
        "batch_size": "Items Per Batch",
        "prev_batch": "Previous Batch",
        "next_batch": "Next Batch",
        "generate_preview": "Generate Audition Audio",
        "preview_list": "## Audition List",
        "ref_audio": "Ref Audio",
        "ref_language": "Ref Text Language",
        "ref_text": "Ref Text",
        "generated": "Audition Audio",
        "save": "Accept",
        "saved": "Saved",
        "save_failed": "Save Failed",
        "version_text": "SoVITS Version: {version}",
        "status_refreshed": "Refreshed. SoVITS: {sovits} | GPT: {gpt}",
        "status_loaded_sovits": "Loaded SoVITS: {name}",
        "status_failed_sovits": "Failed to load SoVITS: {error}",
        "status_loaded_gpt": "Loaded GPT: {name}",
        "status_failed_gpt": "Failed to load GPT: {error}",
        "status_auto_switched_sovits": "Loaded GPT: {gpt}, auto-switched SoVITS: {sovits}",
        "status_auto_switch_sovits_failed": "Loaded GPT: {gpt}, but failed to auto-switch SoVITS: {error}",
        "status_no_paired_sovits": "Loaded GPT: {gpt}, no same-name SoVITS found.",
        "no_model_found": "No model found. Put .pth files into SoVITS_weights* and .ckpt files into GPT_weights*.",
        "lang_zh": "中文",
        "lang_ja": "日本語",
        "lang_en": "English",
        "cut_no_cut": "No Cut",
        "cut_every_4_sentences": "Every 4 Sentences",
        "cut_every_50_chars": "Every 50 Characters",
        "cut_zh_full_stop": "By Chinese Full Stop",
        "cut_en_full_stop": "By English Full Stop",
        "cut_punctuation": "By Punctuation",
    },
}

UI_LANGUAGE_OPTIONS = [("中文", "zh"), ("日本語", "ja"), ("English", "en")]
ROOT_PAIR_MAP = {gpt_root: sovits_root for gpt_root, sovits_root in zip(GPT_WEIGHT_ROOTS, SOVITS_WEIGHT_ROOTS)}

APP_CSS = """
:root {
    color-scheme: dark;
}

body {
    background:
        radial-gradient(circle at top left, rgba(65, 102, 175, 0.22), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 116, 32, 0.12), transparent 24%),
        linear-gradient(180deg, #08111f 0%, #0d1728 100%);
}

.gradio-container {
    max-width: 1920px !important;
    padding: 24px 36px 48px !important;
    background: transparent !important;
    font-family: "Segoe UI", "Microsoft YaHei UI", "Noto Sans SC", sans-serif !important;
    --body-text-color: #f3f7ff;
    --body-text-color-subdued: #92a6c6;
    --background-fill-primary: rgba(8, 15, 27, 0.72);
    --background-fill-secondary: rgba(17, 29, 47, 0.94);
    --block-background-fill: rgba(26, 38, 58, 0.94);
    --block-border-color: rgba(122, 146, 183, 0.18);
    --block-border-width: 1px;
    --block-radius: 18px;
    --block-shadow: 0 14px 36px rgba(0, 0, 0, 0.18);
    --block-padding: 16px;
    --input-background-fill: rgba(21, 34, 53, 0.96);
    --input-border-color: rgba(90, 118, 155, 0.36);
    --input-border-color-focus: rgba(63, 126, 255, 0.8);
    --input-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    --input-shadow-focus:
        0 0 0 1px rgba(63, 126, 255, 0.62),
        0 0 0 4px rgba(63, 126, 255, 0.12);
    --button-secondary-background-fill:
        linear-gradient(180deg, rgba(88, 100, 120, 0.9) 0%, rgba(71, 82, 101, 0.96) 100%);
    --button-secondary-background-fill-hover:
        linear-gradient(180deg, rgba(103, 116, 138, 0.96) 0%, rgba(80, 92, 112, 1) 100%);
    --button-secondary-border-color: rgba(132, 154, 190, 0.16);
    --button-secondary-text-color: #f7f9fc;
    --button-primary-background-fill: linear-gradient(135deg, #ff8b1f 0%, #ff6200 100%);
    --button-primary-background-fill-hover: linear-gradient(135deg, #ff962d 0%, #ff6f12 100%);
    --button-primary-border-color: transparent;
    --button-primary-text-color: #fffaf4;
    --color-accent: #2f78ff;
}

.app-shell {
    gap: 18px !important;
}

.app-title,
.app-subtitle {
    display: none !important;
}

.section-panel {
    padding: 0 !important;
    overflow: hidden !important;
    border: 1px solid rgba(122, 146, 183, 0.18) !important;
    border-radius: 20px !important;
    background: linear-gradient(180deg, rgba(18, 30, 47, 0.96) 0%, rgba(16, 26, 40, 0.96) 100%) !important;
    box-shadow: 0 20px 44px rgba(0, 0, 0, 0.16) !important;
}

.section-heading {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
}

.section-heading h2 {
    margin: 0 !important;
    padding: 14px 22px !important;
    border-bottom: 1px solid rgba(152, 172, 203, 0.12);
    background: linear-gradient(90deg, rgba(76, 89, 110, 0.96) 0%, rgba(64, 75, 94, 0.92) 100%);
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

.control-row {
    flex-wrap: wrap !important;
    gap: 14px !important;
    padding: 18px 22px 0 !important;
}

.control-row > * {
    min-width: 0 !important;
}

.model-meta-row,
.synth-secondary-row,
.batch-row {
    padding-bottom: 22px !important;
}

.model-meta-row,
.synth-secondary-row {
    padding-top: 12px !important;
}

.gradio-container label > span {
    color: #f4f8ff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

.gradio-container input,
.gradio-container textarea {
    font-size: 16px !important;
}

.gradio-container button {
    transition: transform 0.16s ease, filter 0.16s ease !important;
    box-shadow: none !important;
}

.gradio-container button:hover {
    transform: translateY(-1px);
    filter: brightness(1.04);
}

.ui-language-select,
.model-version-box,
.status-box {
    min-height: 100% !important;
}

.status-box input,
.status-box textarea,
.model-version-box input,
.model-version-box textarea {
    font-weight: 600 !important;
}

.test-text textarea {
    min-height: 126px !important;
}

.model-row .refresh-button,
.batch-actions-row .batch-nav,
.batch-actions-row .batch-cta {
    height: 100% !important;
    min-height: 118px !important;
    border-radius: 16px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

.model-row .refresh-button {
    background: linear-gradient(180deg, rgba(93, 105, 125, 0.94) 0%, rgba(74, 85, 104, 0.98) 100%) !important;
}

.batch-actions-row {
    gap: 14px !important;
    height: 100% !important;
}

.batch-actions-row .batch-nav {
    background: linear-gradient(180deg, rgba(88, 100, 120, 0.9) 0%, rgba(71, 82, 101, 0.96) 100%) !important;
}

.batch-actions-row .batch-cta {
    background: linear-gradient(135deg, #ff8b1f 0%, #ff6200 100%) !important;
}

.preview-list-shell {
    gap: 14px !important;
    padding: 18px 22px 22px !important;
}

.preview-item {
    align-items: stretch !important;
    flex-wrap: nowrap !important;
    gap: 14px !important;
    padding: 14px !important;
    border: 1px solid rgba(122, 146, 183, 0.16);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(24, 37, 58, 0.72) 0%, rgba(20, 31, 49, 0.78) 100%);
}

.preview-item > * {
    align-self: stretch !important;
    min-width: 0 !important;
}

.preview-meta {
    gap: 12px !important;
    min-height: 0 !important;
}

.preview-ref-audio,
.preview-gen-audio {
    min-height: 0 !important;
}

.preview-ref-lang input,
.preview-ref-lang textarea {
    min-height: 56px !important;
    max-height: 56px !important;
}

.preview-ref-text textarea {
    min-height: 132px !important;
    max-height: 132px !important;
    overflow-y: auto !important;
}

.preview-save {
    min-height: 0 !important;
    border-radius: 16px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background: linear-gradient(180deg, rgba(93, 105, 125, 0.94) 0%, rgba(74, 85, 104, 0.98) 100%) !important;
}

@media (max-width: 1600px) {
    .preview-item {
        flex-wrap: wrap !important;
    }

    .preview-save {
        min-height: 84px !important;
    }
}

@media (max-width: 1200px) {
    .gradio-container {
        padding: 18px !important;
    }

    .control-row {
        padding: 16px 16px 0 !important;
    }

    .preview-list-shell {
        padding: 16px !important;
    }

    .section-heading h2 {
        padding: 12px 16px !important;
        font-size: 18px !important;
    }

    .model-row .refresh-button,
    .batch-actions-row .batch-nav,
    .batch-actions-row .batch-cta {
        min-height: 86px !important;
        font-size: 18px !important;
    }
}
"""

SYNTH_LANGUAGE_LABELS = {
    "Chinese": {"zh": "中文", "ja": "中国語", "en": "Chinese"},
    "English": {"zh": "英语", "ja": "英語", "en": "English"},
    "Japanese": {"zh": "日语", "ja": "日本語", "en": "Japanese"},
    "Cantonese": {"zh": "粤语", "ja": "広東語", "en": "Cantonese"},
    "Korean": {"zh": "韩语", "ja": "韓国語", "en": "Korean"},
    "Chinese-English Mix": {"zh": "中英混合", "ja": "中英ミックス", "en": "Chinese-English Mix"},
    "Japanese-English Mix": {"zh": "日英混合", "ja": "日英ミックス", "en": "Japanese-English Mix"},
    "Cantonese-English Mix": {"zh": "粤英混合", "ja": "広東語・英語ミックス", "en": "Cantonese-English Mix"},
    "Korean-English Mix": {"zh": "韩英混合", "ja": "韓英ミックス", "en": "Korean-English Mix"},
    "Auto": {"zh": "自动", "ja": "自動", "en": "Auto"},
    "Auto (Cantonese Priority)": {"zh": "自动（粤语优先）", "ja": "自動（広東語優先）", "en": "Auto (Cantonese Priority)"},
}


def custom_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def t(key: str, lang: Optional[str] = None, **kwargs) -> str:
    actual_lang = lang or g_ui_language
    table = UI_TEXTS.get(actual_lang, UI_TEXTS["en"])
    template = table.get(key, UI_TEXTS["en"].get(key, key))
    return template.format(**kwargs) if kwargs else template


def get_cut_method_choices(lang: Optional[str] = None):
    return [(t(f"cut_{item_id}", lang), item_id) for item_id, _ in CUT_METHOD_OPTIONS]


def _translate_synthesis_language(label: str, ui_lang: Optional[str] = None) -> str:
    actual_lang = ui_lang or g_ui_language
    entry = SYNTH_LANGUAGE_LABELS.get(label)
    if not entry:
        return label
    return entry.get(actual_lang, label)


def get_synthesis_language_choices(languages: List[str], ui_lang: Optional[str] = None):
    return [(_translate_synthesis_language(label, ui_lang), label) for label in languages]


def check_audio_duration(path: str) -> bool:
    try:
        wav16k, _ = librosa.load(path, sr=16000)
        return 48000 <= wav16k.shape[0] <= 160000
    except Exception as exc:
        print(f"Error when checking audio {path}: {exc}")
        return False


def remove_noncompliant_audio_from_list() -> None:
    global g_ref_list, g_ref_list_max_index
    print("Checking audio duration ...")
    filtered = []
    for item in tqdm(g_ref_list):
        if check_audio_duration(item["path"]):
            filtered.append(item)
    g_ref_list = filtered
    g_ref_list_max_index = max(len(g_ref_list) - 1, 0)


def load_ref_list_file(path: str) -> None:
    global g_ref_list, g_ref_list_max_index
    records: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            audio_path = row[0].strip()
            if g_ref_folder:
                audio_path = os.path.join(g_ref_folder, os.path.basename(audio_path))

            records.append(
                {
                    "path": audio_path,
                    "lang": row[2].strip(),
                    "text": row[3].strip(),
                }
            )

    g_ref_list = records
    g_ref_list_max_index = max(len(g_ref_list) - 1, 0)


def get_gradio_allowed_paths() -> List[str]:
    allowed = set()
    if g_ref_folder:
        allowed.add(os.path.abspath(g_ref_folder))

    for item in g_ref_list:
        ref_path = item.get("path")
        if not ref_path:
            continue
        allowed.add(os.path.abspath(os.path.dirname(ref_path)))

    return sorted(allowed)


def get_weights_names():
    sovits_names: List[str] = []
    for root in SOVITS_WEIGHT_ROOTS:
        os.makedirs(root, exist_ok=True)
        for name in os.listdir(root):
            if name.lower().endswith(".pth"):
                sovits_names.append(f"{root}/{name}")

    gpt_names: List[str] = []
    for root in GPT_WEIGHT_ROOTS:
        os.makedirs(root, exist_ok=True)
        for name in os.listdir(root):
            if name.lower().endswith(".ckpt"):
                gpt_names.append(f"{root}/{name}")

    return sorted(sovits_names, key=custom_sort_key), sorted(gpt_names, key=custom_sort_key)


def refresh_model_list():
    global g_SoVITS_names, g_GPT_names
    g_SoVITS_names, g_GPT_names = get_weights_names()

    status = t("status_refreshed", sovits=len(g_SoVITS_names), gpt=len(g_GPT_names))
    return (
        {"choices": g_SoVITS_names, "__type__": "update"},
        {"choices": g_GPT_names, "__type__": "update"},
        status,
    )


def _normalize_ref_language(raw_language: Optional[str]) -> str:
    if raw_language is None:
        return inference_main.get_default_language()

    lang = str(raw_language).strip()
    if lang in inference_main.get_supported_languages():
        return lang

    if lang in LEGACY_REF_LANGUAGE_ALIASES:
        candidate = LEGACY_REF_LANGUAGE_ALIASES[lang]
        if candidate in inference_main.get_supported_languages():
            return candidate

    return inference_main.normalize_ref_language(lang)


def reload_data(index: int, batch: int):
    global g_index, g_batch
    g_index = index
    g_batch = batch
    return g_ref_list[index:index + batch]


def change_index(index: int, batch: int):
    global g_index, g_batch, g_ref_audio_path_list

    g_ref_audio_path_list = []
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)

    output = []

    # Reference audio widgets
    for item in datas:
        output.append(
            {
                "__type__": "update",
                "label": f"{t('ref_audio')} {os.path.basename(item['path'])}",
                "value": item["path"],
            }
        )
        g_ref_audio_path_list.append(item["path"])

    for _ in range(g_batch - len(datas)):
        output.append(
            {
                "__type__": "update",
                "label": t("ref_audio"),
                "value": None,
            }
        )
        g_ref_audio_path_list.append(None)

    # Reference language widgets
    for item in datas:
        output.append(_normalize_ref_language(item["lang"]))
    for _ in range(g_batch - len(datas)):
        output.append(None)

    # Reference text widgets
    for item in datas:
        output.append(item["text"])
    for _ in range(g_batch - len(datas)):
        output.append(None)

    # Test audio widgets
    for _ in range(g_batch):
        output.append(None)

    # Save buttons
    for _ in datas:
        output.append({"__type__": "update", "value": t("save"), "interactive": True})
    for _ in range(g_batch - len(datas)):
        output.append({"__type__": "update", "value": t("save"), "interactive": False})

    # Preview row visibility, hide rows without data to avoid blank gaps.
    for _ in datas:
        output.append({"__type__": "update", "visible": True})
    for _ in range(g_batch - len(datas)):
        output.append({"__type__": "update", "visible": False})

    return output


def previous_index(index: int, batch: int):
    if (index - batch) >= 0:
        new_index = index - batch
    else:
        new_index = 0
    return new_index, *change_index(new_index, batch)


def next_index(index: int, batch: int):
    if (index + batch) <= g_ref_list_max_index:
        new_index = index + batch
    else:
        new_index = index
    return new_index, *change_index(new_index, batch)


def copy_proved_ref_audio(index: int, text: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    filename = re.sub(r"[/\\:*?\"<>|]", "", text or "untitled")
    if not filename:
        filename = "untitled"

    try:
        source = g_ref_audio_path_list[int(index)]
        if not source:
            raise FileNotFoundError("Reference audio path is empty.")
        shutil.copy2(source, os.path.join(out_dir, f"{filename}.wav"))
        return {"__type__": "update", "value": t("saved"), "interactive": False}
    except Exception as exc:
        print(exc)
        return {"__type__": "update", "value": t("save_failed"), "interactive": True}


def _build_language_update(current_text_language: Optional[str]):
    languages = inference_main.get_supported_languages()
    if not languages:
        return {"__type__": "update", "choices": [], "value": None}

    value = current_text_language if current_text_language in languages else languages[0]
    return {
        "__type__": "update",
        "choices": get_synthesis_language_choices(languages),
        "value": value,
    }


def _build_version_updates(version: str):
    is_v3 = version == "v3"
    sample_value = 32 if is_v3 else 8

    sample_steps_update = {
        "__type__": "update",
        "visible": is_v3,
        "interactive": is_v3,
        "value": sample_value,
    }

    super_sampling_update = {
        "__type__": "update",
        "visible": is_v3,
        "interactive": is_v3,
        "value": False,
    }

    return (
        t("version_text", version=version),
        sample_steps_update,
        super_sampling_update,
    )


def on_change_sovits_weights(sovits_path: str, current_text_language: str):
    try:
        version = inference_main.change_sovits_weights(sovits_path)
        language_update = _build_language_update(current_text_language)
        version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
        status = t("status_loaded_sovits", name=os.path.basename(sovits_path))
        return (
            language_update,
            version_text,
            sample_steps_update,
            super_sampling_update,
            status,
        )
    except Exception as exc:
        print(exc)
        version = inference_main.get_current_model_version()
        language_update = _build_language_update(current_text_language)
        version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
        return (
            language_update,
            version_text,
            sample_steps_update,
            super_sampling_update,
            t("status_failed_sovits", error=exc),
        )


def _find_paired_sovits_path(gpt_path: str) -> Optional[str]:
    if not gpt_path:
        return None

    gpt_root = os.path.normpath(gpt_path).split(os.sep)[0]
    preferred_roots: List[str] = []
    paired_root = ROOT_PAIR_MAP.get(gpt_root)
    if paired_root:
        preferred_roots.append(paired_root)

    for root in SOVITS_WEIGHT_ROOTS:
        if root not in preferred_roots:
            preferred_roots.append(root)

    stem = os.path.splitext(os.path.basename(gpt_path))[0]
    for root in preferred_roots:
        candidate = f"{root}/{stem}.pth"
        if candidate in g_SoVITS_names:
            return candidate
    return None


def on_change_gpt_weights(gpt_path: str, current_sovits_path: str, current_text_language: str):
    try:
        inference_main.change_gpt_weights(gpt_path)
    except Exception as exc:
        print(exc)
        version = inference_main.get_current_model_version()
        language_update = _build_language_update(current_text_language)
        version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
        return (
            {"__type__": "update"},
            language_update,
            version_text,
            sample_steps_update,
            super_sampling_update,
            t("status_failed_gpt", error=exc),
        )

    gpt_name = os.path.basename(gpt_path)
    paired_sovits = _find_paired_sovits_path(gpt_path)
    if paired_sovits and paired_sovits != current_sovits_path:
        try:
            version = inference_main.change_sovits_weights(paired_sovits)
            language_update = _build_language_update(current_text_language)
            version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
            return (
                {"__type__": "update", "value": paired_sovits},
                language_update,
                version_text,
                sample_steps_update,
                super_sampling_update,
                t(
                    "status_auto_switched_sovits",
                    gpt=gpt_name,
                    sovits=os.path.basename(paired_sovits),
                ),
            )
        except Exception as exc:
            print(exc)
            version = inference_main.get_current_model_version()
            language_update = _build_language_update(current_text_language)
            version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
            return (
                {"__type__": "update"},
                language_update,
                version_text,
                sample_steps_update,
                super_sampling_update,
                t("status_auto_switch_sovits_failed", gpt=gpt_name, error=exc),
            )

    version = inference_main.get_current_model_version()
    language_update = _build_language_update(current_text_language)
    version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
    status_key = "status_loaded_gpt" if paired_sovits else "status_no_paired_sovits"
    return (
        {"__type__": "update"},
        language_update,
        version_text,
        sample_steps_update,
        super_sampling_update,
        t(status_key, name=gpt_name, gpt=gpt_name),
    )


def generate_test_audio(
    test_text: str,
    text_language: str,
    cut_method_id: str,
    top_k: float,
    top_p: float,
    temperature: float,
    speed_factor: float,
    repetition_penalty: float,
    seed: float,
    sample_steps: float,
    super_sampling: bool,
    *widgets,
):
    output = []

    if not test_text or not test_text.strip():
        return [None for _ in range(g_batch)]

    cut_mode = CUT_METHOD_VALUE_BY_ID.get(cut_method_id, 0)

    for i in range(g_batch):
        ref_audio_path = g_ref_audio_path_list[i]
        if not ref_audio_path:
            output.append(None)
            continue

        ref_lang = widgets[i]
        ref_text = widgets[i + g_batch]

        try:
            generated = inference_main.get_tts_wav(
                ref_audio_path,
                ref_text or "",
                ref_lang,
                test_text,
                text_language,
                how_to_cut=cut_mode,
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                speed_factor=float(speed_factor),
                repetition_penalty=float(repetition_penalty),
                seed=int(seed),
                sample_steps=int(sample_steps),
                super_sampling=bool(super_sampling),
            )
            sample_rate, audio_array = next(generated)
            output.append((sample_rate, audio_array))
        except Exception as exc:
            print(f"Skip {ref_audio_path}: {exc}")
            output.append(None)

    return output


def change_ui_language(ui_lang: str, current_cut_method: str, current_text_language: str):
    global g_ui_language
    if ui_lang not in UI_TEXTS:
        ui_lang = "en"
    g_ui_language = ui_lang

    version_text, _, _ = _build_version_updates(inference_main.get_current_model_version())
    cut_choices = get_cut_method_choices(ui_lang)
    cut_values = [item_id for item_id, _ in CUT_METHOD_OPTIONS]
    cut_value = current_cut_method if current_cut_method in cut_values else "every_4_sentences"
    supported_languages = inference_main.get_supported_languages()
    text_language_value = (
        current_text_language
        if current_text_language in supported_languages
        else (supported_languages[0] if supported_languages else None)
    )

    output = [
        t("title", ui_lang),
        t("subtitle", ui_lang),
        {"__type__": "update", "label": t("ui_language", ui_lang)},
        t("model_selection", ui_lang),
        {"__type__": "update", "label": t("gpt_model", ui_lang)},
        {"__type__": "update", "label": t("sovits_model", ui_lang)},
        {"__type__": "update", "label": t("output_dir", ui_lang)},
        {"__type__": "update", "value": t("refresh_models", ui_lang)},
        {"__type__": "update", "label": t("model_version", ui_lang), "value": version_text},
        {"__type__": "update", "label": t("status", ui_lang), "value": t("ready", ui_lang)},
        t("synthesis_options", ui_lang),
        {"__type__": "update", "label": t("preview_text", ui_lang), "placeholder": t("preview_placeholder", ui_lang)},
        {
            "__type__": "update",
            "label": t("synthesis_language", ui_lang),
            "choices": get_synthesis_language_choices(supported_languages, ui_lang),
            "value": text_language_value,
        },
        {"__type__": "update", "label": t("split_method", ui_lang), "choices": cut_choices, "value": cut_value},
        {"__type__": "update", "label": t("sample_steps_v3", ui_lang)},
        {"__type__": "update", "label": t("super_sampling_v3", ui_lang)},
        t("batch_preview", ui_lang),
        {"__type__": "update", "label": t("start_index", ui_lang)},
        {"__type__": "update", "label": t("batch_size", ui_lang)},
        {"__type__": "update", "value": t("prev_batch", ui_lang)},
        {"__type__": "update", "value": t("next_batch", ui_lang)},
        {"__type__": "update", "value": t("generate_preview", ui_lang)},
        t("preview_list", ui_lang),
    ]

    for i in range(g_batch):
        ref_audio_path = g_ref_audio_path_list[i] if i < len(g_ref_audio_path_list) else None
        label = t("ref_audio", ui_lang)
        if ref_audio_path:
            label = f"{label} {os.path.basename(ref_audio_path)}"
        output.append({"__type__": "update", "label": label})

    for _ in range(g_batch):
        output.append({"__type__": "update", "label": t("ref_language", ui_lang)})
    for _ in range(g_batch):
        output.append({"__type__": "update", "label": t("ref_text", ui_lang)})
    for _ in range(g_batch):
        output.append({"__type__": "update", "label": t("generated", ui_lang)})
    for _ in range(g_batch):
        output.append({"__type__": "update", "value": t("save", ui_lang)})

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", type=str, default="ref.list", help="Reference list file path")
    parser.add_argument("-p", "--port", type=int, default=14285, help="WebUI port")
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="",
        help="Optional folder for reference audio; basename in list will be joined to this folder.",
    )
    parser.add_argument("-b", "--batch", type=int, default=10, help="Batch size per page")
    parser.add_argument(
        "-cd",
        "--check_duration",
        action="store_true",
        default=False,
        help="Check whether each reference audio duration is between 3 and 10 seconds.",
    )
    parser.add_argument(
        "-r",
        "--random_order",
        action="store_true",
        default=False,
        help="Randomize the order of reference audios.",
    )
    args = parser.parse_args()

    g_ref_audio_widget_list = []
    g_ref_lang_widget_list = []
    g_ref_text_widget_list = []
    g_test_audio_widget_list = []
    g_save_widget_list = []
    g_preview_row_widget_list = []

    g_ref_folder = args.folder
    g_batch = args.batch

    load_ref_list_file(args.list)

    if args.check_duration:
        remove_noncompliant_audio_from_list()

    if args.random_order:
        random.shuffle(g_ref_list)

    g_SoVITS_names, g_GPT_names = get_weights_names()

    if not g_GPT_names or not g_SoVITS_names:
        print(t("no_model_found"))
        raise SystemExit(1)

    current_version = inference_main.initialize(g_GPT_names[0], g_SoVITS_names[0])
    language_choices = inference_main.get_supported_languages()
    default_language = language_choices[0] if language_choices else inference_main.get_default_language()
    version_text = t("version_text", version=current_version)
    is_v3 = current_version == "v3"

    with gr.Blocks(title="GPT-SoVITS RefAudio Tester WebUI", css=APP_CSS) as app:
        with gr.Column(elem_classes=["app-shell"]):
            mdTitle = gr.Markdown(t("title"), elem_classes=["app-title"])
            mdSubtitle = gr.Markdown(t("subtitle"), elem_classes=["app-subtitle"])

            with gr.Group(elem_classes=["section-panel", "model-section"]):
                mdModelSelection = gr.Markdown(t("model_selection"), elem_classes=["section-heading"])
                with gr.Row(elem_classes=["control-row", "model-row"], equal_height=True):
                    dropdownGPT = gr.Dropdown(
                        label=t("gpt_model"),
                        choices=g_GPT_names,
                        value=g_GPT_names[0],
                        interactive=True,
                        scale=3,
                    )
                    dropdownSoVITS = gr.Dropdown(
                        label=t("sovits_model"),
                        choices=g_SoVITS_names,
                        value=g_SoVITS_names[0],
                        interactive=True,
                        scale=3,
                    )
                    textboxOutputFolder = gr.Textbox(
                        label=t("output_dir"),
                        value="output/",
                        interactive=True,
                        scale=3,
                    )
                    btnRefresh = gr.Button(
                        t("refresh_models"),
                        scale=2,
                        elem_classes=["refresh-button"],
                    )

                with gr.Row(elem_classes=["control-row", "model-meta-row"], equal_height=True):
                    dropdownUILanguage = gr.Dropdown(
                        label=t("ui_language"),
                        choices=UI_LANGUAGE_OPTIONS,
                        value=g_ui_language,
                        interactive=True,
                        scale=2,
                        elem_classes=["ui-language-select"],
                    )
                    textboxModelVersion = gr.Textbox(
                        label=t("model_version"),
                        value=version_text,
                        interactive=False,
                        scale=3,
                        elem_classes=["model-version-box"],
                    )
                    textboxStatus = gr.Textbox(
                        label=t("status"),
                        value=t("ready"),
                        interactive=False,
                        scale=5,
                        elem_classes=["status-box"],
                    )

            with gr.Group(elem_classes=["section-panel", "synthesis-section"]):
                mdSynthesisOptions = gr.Markdown(t("synthesis_options"), elem_classes=["section-heading"])
                with gr.Row(elem_classes=["control-row", "synth-primary-row"], equal_height=True):
                    textboxTestText = gr.Textbox(
                        label=t("preview_text"),
                        interactive=True,
                        placeholder=t("preview_placeholder"),
                        lines=3,
                        max_lines=4,
                        scale=5,
                        elem_classes=["test-text"],
                    )
                    dropdownTextLanguage = gr.Dropdown(
                        label=t("synthesis_language"),
                        choices=get_synthesis_language_choices(language_choices),
                        value=default_language,
                        interactive=True,
                        scale=2,
                    )
                    dropdownHowToCut = gr.Dropdown(
                        label=t("split_method"),
                        choices=get_cut_method_choices(),
                        value="every_4_sentences",
                        interactive=True,
                        scale=2,
                    )
                    sliderTopK = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label="top_k",
                        value=20,
                        interactive=True,
                        scale=2,
                    )
                    sliderTopP = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        label="top_p",
                        value=0.6,
                        interactive=True,
                        scale=2,
                    )
                    sliderTemperature = gr.Slider(
                        minimum=0,
                        maximum=1.5,
                        step=0.01,
                        label="temperature",
                        value=0.6,
                        interactive=True,
                        scale=2,
                    )

                with gr.Row(elem_classes=["control-row", "synth-secondary-row"], equal_height=True):
                    sliderSpeedFactor = gr.Slider(
                        minimum=0.6,
                        maximum=1.65,
                        step=0.01,
                        label="speed_factor",
                        value=1.0,
                        interactive=True,
                        scale=2,
                    )
                    sliderRepetitionPenalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.01,
                        label="repetition_penalty",
                        value=1.35,
                        interactive=True,
                        scale=2,
                    )
                    numberSeed = gr.Number(
                        label="seed (-1 random)",
                        value=-1,
                        precision=0,
                        interactive=True,
                        scale=2,
                    )
                    sliderSampleSteps = gr.Slider(
                        minimum=4,
                        maximum=128,
                        step=4,
                        label=t("sample_steps_v3"),
                        value=32 if is_v3 else 8,
                        visible=is_v3,
                        interactive=is_v3,
                        scale=2,
                    )
                    checkboxSuperSampling = gr.Checkbox(
                        label=t("super_sampling_v3"),
                        value=False,
                        visible=is_v3,
                        interactive=is_v3,
                        scale=1,
                    )

            with gr.Group(elem_classes=["section-panel", "batch-section"]):
                mdBatchPreview = gr.Markdown(t("batch_preview"), elem_classes=["section-heading"])
                with gr.Row(elem_classes=["control-row", "batch-row"], equal_height=True):
                    with gr.Column(scale=6):
                        with gr.Row(elem_classes=["batch-sliders"], equal_height=True):
                            sliderStartIndex = gr.Slider(
                                minimum=0,
                                maximum=g_ref_list_max_index,
                                step=max(g_batch, 1),
                                label=t("start_index"),
                                value=0,
                                interactive=True,
                                scale=4,
                            )
                            sliderBatchSize = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                label=t("batch_size"),
                                value=g_batch,
                                interactive=False,
                                scale=3,
                            )

                    with gr.Column(scale=7):
                        with gr.Row(elem_classes=["batch-actions-row"], equal_height=True):
                            btnPreBatch = gr.Button(
                                t("prev_batch"),
                                scale=1,
                                elem_classes=["batch-nav"],
                            )
                            btnNextBatch = gr.Button(
                                t("next_batch"),
                                scale=1,
                                elem_classes=["batch-nav"],
                            )
                            btnInference = gr.Button(
                                t("generate_preview"),
                                variant="primary",
                                scale=2,
                                elem_classes=["batch-cta"],
                            )

            with gr.Group(elem_classes=["section-panel", "preview-section"]):
                mdPreviewList = gr.Markdown(t("preview_list"), elem_classes=["section-heading"])
                with gr.Column(elem_classes=["preview-list-shell"]):
                    for i in range(g_batch):
                        with gr.Row(elem_classes=["preview-item"], equal_height=True) as preview_row:
                            ref_no = gr.Number(value=i, visible=False)
                            ref_audio = gr.Audio(
                                label=t("ref_audio"),
                                visible=True,
                                scale=5,
                                min_width=280,
                                elem_classes=["preview-ref-audio"],
                            )
                            with gr.Column(scale=5, elem_classes=["preview-meta"]):
                                ref_lang = gr.Textbox(
                                    label=t("ref_language"),
                                    visible=True,
                                    min_width=140,
                                    lines=1,
                                    max_lines=1,
                                    elem_classes=["preview-ref-lang"],
                                )
                                ref_text = gr.Textbox(
                                    label=t("ref_text"),
                                    visible=True,
                                    min_width=280,
                                    lines=3,
                                    max_lines=3,
                                    elem_classes=["preview-ref-text"],
                                )
                            test_audio = gr.Audio(
                                label=t("generated"),
                                visible=True,
                                scale=5,
                                min_width=280,
                                elem_classes=["preview-gen-audio"],
                            )
                            save = gr.Button(
                                value=t("save"),
                                scale=1,
                                min_width=120,
                                elem_classes=["preview-save"],
                            )

                            g_ref_audio_widget_list.append(ref_audio)
                            g_ref_lang_widget_list.append(ref_lang)
                            g_ref_text_widget_list.append(ref_text)
                            g_test_audio_widget_list.append(test_audio)
                            g_save_widget_list.append(save)
                            g_preview_row_widget_list.append(preview_row)

                            save.click(
                                copy_proved_ref_audio,
                                inputs=[ref_no, ref_text, textboxOutputFolder],
                                outputs=[save],
                            )

            btnRefresh.click(
                fn=refresh_model_list,
                inputs=[],
                outputs=[dropdownSoVITS, dropdownGPT, textboxStatus],
            )

            dropdownSoVITS.change(
                on_change_sovits_weights,
                inputs=[dropdownSoVITS, dropdownTextLanguage],
                outputs=[
                    dropdownTextLanguage,
                    textboxModelVersion,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    textboxStatus,
                ],
            )

            dropdownGPT.change(
                on_change_gpt_weights,
                inputs=[dropdownGPT, dropdownSoVITS, dropdownTextLanguage],
                outputs=[
                    dropdownSoVITS,
                    dropdownTextLanguage,
                    textboxModelVersion,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    textboxStatus,
                ],
            )

            dropdownUILanguage.change(
                change_ui_language,
                inputs=[dropdownUILanguage, dropdownHowToCut, dropdownTextLanguage],
                outputs=[
                    mdTitle,
                    mdSubtitle,
                    dropdownUILanguage,
                    mdModelSelection,
                    dropdownGPT,
                    dropdownSoVITS,
                    textboxOutputFolder,
                    btnRefresh,
                    textboxModelVersion,
                    textboxStatus,
                    mdSynthesisOptions,
                    textboxTestText,
                    dropdownTextLanguage,
                    dropdownHowToCut,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    mdBatchPreview,
                    sliderStartIndex,
                    sliderBatchSize,
                    btnPreBatch,
                    btnNextBatch,
                    btnInference,
                    mdPreviewList,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                ],
            )

            sliderStartIndex.change(
                change_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                    *g_preview_row_widget_list,
                ],
            )

            btnPreBatch.click(
                previous_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                    *g_preview_row_widget_list,
                ],
            )

            btnNextBatch.click(
                next_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                    *g_preview_row_widget_list,
                ],
            )

            btnInference.click(
                generate_test_audio,
                inputs=[
                    textboxTestText,
                    dropdownTextLanguage,
                    dropdownHowToCut,
                    sliderTopK,
                    sliderTopP,
                    sliderTemperature,
                    sliderSpeedFactor,
                    sliderRepetitionPenalty,
                    numberSeed,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                ],
                outputs=[*g_test_audio_widget_list],
            )

            app.load(
                change_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                    *g_preview_row_widget_list,
                ],
            )

    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        quiet=True,
        share=False,
        server_port=args.port,
        allowed_paths=get_gradio_allowed_paths(),
    )
