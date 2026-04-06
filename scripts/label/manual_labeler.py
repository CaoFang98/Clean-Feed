#!/usr/bin/env python3
"""
任务化人工打标工具：
对每条样本分别标注多个二分类任务，而不是只打一个互斥总标签。
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from task_config import DEFAULT_TASK_ORDER, TASK_DEFINITIONS, build_task_annotation

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "zhihu_cleaned_data.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "task_labeled_dataset.jsonl"
DEFAULT_PRELABEL_PATH = PROJECT_ROOT / "data" / "task_prelabeled_dataset.jsonl"
ANNOTATOR_ID = os.getenv("CLEANFEED_ANNOTATOR_ID", "human_default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanFeed 任务化人工打标工具")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prelabel-path", type=Path, default=DEFAULT_PRELABEL_PATH)
    return parser.parse_args()


def load_all_data(input_path: Path) -> list[dict]:
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def load_labeled_data(output_path: Path) -> list[dict]:
    if not output_path.exists():
        return []
    with open(output_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def build_labeled_index(items: list[dict]) -> dict[str, dict]:
    labeled_index: dict[str, dict] = {}
    for item in items:
        sample_id = item.get("sample_id")
        if sample_id:
            labeled_index[sample_id] = item
    return labeled_index


def load_prelabeled_data(prelabel_path: Path) -> dict[str, dict]:
    if not prelabel_path.exists():
        return {}

    prelabel_map: dict[str, dict] = {}
    with open(prelabel_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            sample_id = item.get("sample_id")
            if sample_id:
                prelabel_map[sample_id] = item
    return prelabel_map


def save_labeled_item(item: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def delete_last_labeled_item(output_path: Path):
    if not output_path.exists():
        return
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines:
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])


def clear_screen():
    os.system("clear")


def format_predicted_annotation(task_id: str, predicted: dict | None) -> str | None:
    if not predicted:
        return None

    label = predicted.get("label")
    if label is True:
        label_text = "是"
    elif label is False:
        label_text = "否"
    else:
        label_text = "不确定"

    confidence = predicted.get("confidence")
    confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "-"
    reason = predicted.get("reason") or "-"
    return f"预标建议: {label_text} | confidence={confidence_text} | reason={reason}"


def can_accept_prediction(predicted: dict | None) -> bool:
    return bool(predicted) and predicted.get("label") in {True, False}


def ask_task_label(task_id: str, predicted: dict | None = None) -> dict:
    task_def = TASK_DEFINITIONS[task_id]

    print(f"\n📌 任务: {task_def['name']}")
    print(f"   定义: {task_def['description']}")
    predicted_line = format_predicted_annotation(task_id, predicted)
    accept_prediction = can_accept_prediction(predicted)
    if predicted_line and accept_prediction:
        print(f"   {predicted_line}")
        print("   输入 a = 接受预标, y = 是, n = 否, s = 跳过/不确定, q = 退出")
    elif predicted_line:
        print(f"   {predicted_line}")
        print("   输入 y = 是, n = 否, s = 跳过/不确定, q = 退出")
    else:
        print("   输入 y = 是, n = 否, s = 跳过/不确定, q = 退出")

    while True:
        choice = input("   你的判断: ").strip().lower()
        if choice == "q":
            raise KeyboardInterrupt
        valid_choices = {"y", "n", "s"}
        if accept_prediction:
            valid_choices.add("a")
        if choice in valid_choices:
            break
        if accept_prediction:
            print("   ❌ 请输入 a / y / n / s / q")
        else:
            print("   ❌ 请输入 y / n / s / q")

    if choice == "a" and accept_prediction:
        return build_task_annotation(
            predicted.get("label"),
            reason=predicted.get("reason", ""),
            annotator=ANNOTATOR_ID,
            source="human_reviewed_auto",
            confidence=1.0,
        )

    if choice == "s":
        return build_task_annotation(
            None,
            annotator=ANNOTATOR_ID,
            source="human",
        )

    label = choice == "y"
    reason = ""
    if label:
        print("   原因选项:")
        for idx, option in enumerate(task_def["reason_options"], start=1):
            print(f"   {idx}. {option}")
        print("   0. 不填原因")

        while True:
            raw = input("   请选择原因编号: ").strip()
            if raw == "0":
                break
            if raw.isdigit() and 1 <= int(raw) <= len(task_def["reason_options"]):
                reason = task_def["reason_options"][int(raw) - 1]
                break
            print("   ❌ 原因编号无效")

    return build_task_annotation(
        label,
        reason=reason,
        annotator=ANNOTATOR_ID,
        source="human",
        confidence=1.0,
    )


def main():
    args = parse_args()
    clear_screen()

    all_data = load_all_data(args.input_path)
    labeled_data = load_labeled_data(args.output_path)
    labeled_index = build_labeled_index(labeled_data)
    prelabel_map = load_prelabeled_data(args.prelabel_path)
    pending_items = [item for item in all_data if item.get("sample_id") not in labeled_index]
    initial_labeled_count = len(labeled_data)
    history: list[dict] = []

    print("=" * 88)
    print("CleanFeed 任务化人工打标工具")
    print("=" * 88)
    print(f"总数据量: {len(all_data)} 条")
    print(f"已打标: {len(labeled_data)} 条")
    print(f"剩余待打标: {len(pending_items)} 条")
    print(f"标注人: {ANNOTATOR_ID}")
    print(f"可用预标建议: {len(prelabel_map)} 条")
    print(f"输入文件: {args.input_path}")
    print(f"输出文件: {args.output_path}")
    print(f"预标文件: {args.prelabel_path}")
    print("\n任务列表:")
    for task_id in DEFAULT_TASK_ORDER:
        task_def = TASK_DEFINITIONS[task_id]
        print(f"  - {task_id}: {task_def['name']}")
    print("\n通用操作:")
    print("  u = 撤销上一条")
    print("  q = 保存并退出")
    print("=" * 88)
    input("\n按回车开始...")

    try:
        current_index = 0
        while current_index < len(pending_items):
            item = pending_items[current_index]
            sample_id = item.get("sample_id", "")
            predicted_item = prelabel_map.get(sample_id, {})
            clear_screen()

            processed_count = initial_labeled_count + current_index
            progress = processed_count / len(all_data) * 100 if all_data else 100.0
            print(f"📝 待标第 {current_index + 1}/{len(pending_items)} 条 | 总进度: {progress:.1f}%")
            print("=" * 88)
            print(f"🆔 sample_id: {sample_id}")
            print(f"❓ 问题: {item.get('question', '')}")
            print(f"\n🧩 插件推理输入:\n{item.get('model_input', item.get('content', ''))[:1200]}")
            print(f"\n👀 回答预览:\n{item.get('answer_preview', '')[:800]}")
            full_context = item.get("answer_full_clean", "")
            if full_context and full_context != item.get("answer_preview", ""):
                print(f"\n📚 完整回答参考:\n{full_context[:1800]}")
            predicted_tasks = predicted_item.get("predicted_task_annotations", {})
            if predicted_tasks:
                print("\n🤖 预标建议:")
                for task_id in DEFAULT_TASK_ORDER:
                    predicted = predicted_tasks.get(task_id)
                    predicted_line = format_predicted_annotation(task_id, predicted)
                    if predicted_line:
                        print(f"  - {task_id}: {predicted_line}")
            print("\n" + "=" * 88)

            # 允许整条样本层级的撤销/退出
            action = input("回车继续标注，输入 u 撤销上一条，输入 q 退出: ").strip().lower()
            if action == "q":
                print(f"\n✅ 已保存 {len(history)} 条本次标注，下次会按 sample_id 自动跳过已完成样本")
                return
            if action == "u":
                if not history:
                    print("❌ 没有可撤销的记录")
                    input("按回车继续...")
                    continue
                delete_last_labeled_item(args.output_path)
                removed = history.pop()
                labeled_data.pop()
                labeled_index.pop(removed.get("sample_id", ""), None)
                current_index -= 1
                print(f"✅ 已撤销，回到待标第 {current_index + 1} 条")
                input("按回车继续...")
                continue

            task_annotations = {}
            for task_id in DEFAULT_TASK_ORDER:
                task_annotations[task_id] = ask_task_label(task_id, predicted_tasks.get(task_id))

            labeled_item = {
                "sample_id": sample_id,
                "text": item.get("model_input", item.get("content", "")),
                "model_input": item.get("model_input", item.get("content", "")),
                "question": item.get("question", ""),
                "answer_preview": item.get("answer_preview", ""),
                "full_context": item.get("answer_full_clean", ""),
                "platform": item.get("platform", ""),
                "url": item.get("url", ""),
                "task_annotations": task_annotations,
            }
            if predicted_tasks:
                labeled_item["predicted_task_annotations"] = predicted_tasks

            save_labeled_item(labeled_item, args.output_path)
            history.append(labeled_item)
            labeled_data.append(labeled_item)
            labeled_index[sample_id] = labeled_item
            current_index += 1

        clear_screen()
        print("🎉 所有样本都已完成任务化标注")
        print(f"结果已保存到: {args.output_path}")

    except KeyboardInterrupt:
        print(f"\n\n✅ 已保存 {len(history)} 条本次标注，下次会按 sample_id 自动跳过已完成样本")


if __name__ == "__main__":
    main()
