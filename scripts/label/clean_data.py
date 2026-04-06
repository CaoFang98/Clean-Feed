#!/usr/bin/env python3
import html
import hashlib
import json
import re
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from task_config import build_empty_task_annotations

# 配置路径
INPUT_PATH = PROJECT_ROOT / "data" / "zhihu_raw_data.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "zhihu_cleaned_data.jsonl"

# 对齐策略
PREVIEW_MAX_CHARS = 200
MIN_ANSWER_CHARS = 20

# 这里只做“轻清洗”，保留低质量/广告/AI 的原始信号
HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.DOTALL)
WHITESPACE_PATTERN = re.compile(r"\s+")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
CONTACT_HINT_PATTERN = re.compile(
    r"(关注我|公众号|微信|vx|加群|领资料|更多干货|求赞|求关注|转载自|本文首发于)",
    re.IGNORECASE,
)

stats = {
    "total": 0,
    "written": 0,
    "filtered_short": 0,
    "html_touched": 0,
    "whitespace_normalized": 0,
    "contact_hint_hits": 0,
    "preview_truncated": 0,
    "preview_changed_vs_legacy": 0,
    "examples": [],
}


def normalize_text(text: str) -> tuple[str, dict]:
    """只做最小必要规范化：解 HTML 实体、去标签、压缩空白。"""
    original = text or ""
    unescaped = html.unescape(original)
    no_tags = HTML_TAG_PATTERN.sub(" ", unescaped)
    no_zero_width = ZERO_WIDTH_PATTERN.sub("", no_tags)
    normalized = WHITESPACE_PATTERN.sub(" ", no_zero_width).strip()

    meta = {
        "had_html": no_tags != unescaped,
        "whitespace_normalized": normalized != no_zero_width,
        "has_contact_hint": bool(CONTACT_HINT_PATTERN.search(normalized)),
        "chars_before": len(original),
        "chars_after": len(normalized),
    }
    return normalized, meta


def build_preview(answer_full_clean: str) -> tuple[str, bool]:
    if len(answer_full_clean) <= PREVIEW_MAX_CHARS:
        return answer_full_clean, False
    return answer_full_clean[:PREVIEW_MAX_CHARS].rstrip() + "...", True


def build_model_input(question: str, answer_preview: str) -> str:
    if question and answer_preview:
        return f"问题：{question}\n回答预览：{answer_preview}"
    if question:
        return f"问题：{question}"
    return answer_preview


def build_sample_id(item: dict, question: str) -> str:
    """优先依赖原始事实字段，避免随 preview 或 collected_at 波动。"""
    stable_parts = [
        item.get("platform", "zhihu"),
        item.get("url", ""),
        item.get("author", ""),
        question,
    ]
    if any(stable_parts[1:]):
        source_key = "||".join(stable_parts)
    else:
        source_key = "||".join(stable_parts + [item.get("answer_full", "") or ""])
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]


def record_example(raw_item: dict, answer_full_clean: str, answer_preview: str, legacy_preview: str):
    if len(stats["examples"]) >= 5:
        return

    stats["examples"].append({
        "question": raw_item.get("question", "")[:60],
        "raw_answer": raw_item.get("answer_full", "")[:160],
        "clean_full": answer_full_clean[:160],
        "legacy_preview": legacy_preview[:120],
        "new_preview": answer_preview[:120],
    })


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            stats["total"] += 1
            item = json.loads(line)

            question, _ = normalize_text(item.get("question", ""))
            question_detail, _ = normalize_text(item.get("question_detail", ""))
            answer_full_clean, answer_meta = normalize_text(item.get("answer_full", ""))
            legacy_preview_clean, _ = normalize_text(item.get("answer_truncated", ""))

            if answer_meta["had_html"]:
                stats["html_touched"] += 1
            if answer_meta["whitespace_normalized"]:
                stats["whitespace_normalized"] += 1
            if answer_meta["has_contact_hint"]:
                stats["contact_hint_hits"] += 1

            if len(answer_full_clean) < MIN_ANSWER_CHARS:
                stats["filtered_short"] += 1
                continue

            answer_preview, preview_was_truncated = build_preview(answer_full_clean)
            model_input = build_model_input(question, answer_preview)
            sample_id = build_sample_id(item, question)

            if preview_was_truncated:
                stats["preview_truncated"] += 1
            if legacy_preview_clean and legacy_preview_clean != answer_preview:
                stats["preview_changed_vs_legacy"] += 1
                record_example(item, answer_full_clean, answer_preview, legacy_preview_clean)

            cleaned_item = {
                "sample_id": sample_id,
                "question": question,
                "question_detail": question_detail,
                "content": model_input,
                "model_input": model_input,
                "answer_preview": answer_preview,
                "answer_full_clean": answer_full_clean,
                "task_annotations": build_empty_task_annotations(),
                "author": item.get("author", ""),
                "votes": item.get("votes", 0),
                "comment_count": item.get("comment_count", 0),
                "platform": item.get("platform", "zhihu"),
                "url": item.get("url", ""),
                "collected_at": item.get("collected_at", ""),
                "extra": {
                    "preview_was_truncated": preview_was_truncated,
                    "preview_max_chars": PREVIEW_MAX_CHARS,
                    "had_html": answer_meta["had_html"],
                    "has_spam": answer_meta["has_contact_hint"],
                    "has_contact_hint": answer_meta["has_contact_hint"],
                    "source_answer_chars": answer_meta["chars_before"],
                    "clean_answer_chars": answer_meta["chars_after"],
                },
            }

            f_out.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print("=== 数据派生完成 ===")
    print(f"输入条数: {stats['total']}")
    print(f"输出条数: {stats['written']}")
    print(f"过滤短内容: {stats['filtered_short']}")
    print(f"去 HTML/标签: {stats['html_touched']}")
    print(f"空白规范化: {stats['whitespace_normalized']}")
    print(f"命中联系/引流提示词: {stats['contact_hint_hits']}")
    print(f"预览被截断: {stats['preview_truncated']}")
    print(f"新预览和旧 answer_truncated 不同: {stats['preview_changed_vs_legacy']}")
    print("\n说明:")
    print("1. content/model_input 是训练与插件推理共用的规范输入")
    print("2. answer_preview 是插件应看到的回答预览")
    print("3. answer_full_clean 仅供人工标注参考，不直接作为推理输入")
    print("\n样例:")
    for i, example in enumerate(stats["examples"], 1):
        print(f"\n样例{i}: {example['question']}")
        print(f"  原始回答: {example['raw_answer']}")
        print(f"  清洗后完整回答: {example['clean_full']}")
        print(f"  旧预览: {example['legacy_preview']}")
        print(f"  新预览: {example['new_preview']}")


if __name__ == "__main__":
    main()
