"""评测运行器

一键运行完整评测流程：
  生成数据集 → 跑 RAG → 评测 → 出报告

用法：
  python -m app.evaluation.runner
  python -m app.evaluation.runner --skip-generate  # 使用已有数据集
"""

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from app.evaluation.dataset import generate_qa_pairs_from_docs, build_ragas_dataset, QAPair
from app.evaluation.metrics import run_evaluation, print_report


async def main():
    parser = argparse.ArgumentParser(description="SuperBizAgent RAG 评测")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="跳过数据集生成，使用已有文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成数据集，不跑评测",
    )
    parser.add_argument(
        "--docs-dir",
        default="aiops-docs",
        help="知识库文档目录（默认 aiops-docs）",
    )
    parser.add_argument(
        "--dataset-dir",
        default="app/evaluation/datasets",
        help="数据集输出目录",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_file = dataset_dir / "qa_dataset.json"

    # ── 第一步：生成/加载评测数据集 ──
    if args.skip_generate and dataset_file.exists():
        logger.info(f"加载已有数据集: {dataset_file}")
        with open(dataset_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        qa_pairs = [
            QAPair(
                question=d["question"],
                ground_truth=d["ground_truth"],
                context=d.get("context", ""),
                source_file=d.get("source_file", ""),
            )
            for d in qa_data
        ]
    else:
        logger.info("第一步：从知识库自动生成评测数据集...")
        qa_pairs = generate_qa_pairs_from_docs(
            docs_dir=args.docs_dir,
            output_path=str(dataset_dir),
        )
        if not qa_pairs:
            logger.error("数据集生成失败，请检查 LLM API 连接")
            return

    print()
    print(f"  评测数据集: {len(qa_pairs)} 个 QA 对")
    for i, pair in enumerate(qa_pairs, 1):
        print(f"    Q{i}: {pair.question[:60]}...")
        print(f"    来源: {pair.source_file}")
        print()

    if args.dry_run:
        print("  --dry-run 模式，跳过评测")
        return

    # ── 第二步：用真实 RAG 流程构建 RAGAS Dataset ──
    logger.info("第二步：通过 RAG 流程生成 answers 和 contexts...")
    ragas_dataset = build_ragas_dataset(qa_pairs, session_id="eval_session")

    if len(ragas_dataset) == 0:
        logger.error("RAGAS Dataset 为空，无法继续评测")
        return

    # ── 第三步：RAGAS 多维度评测 ──
    logger.info(f"第三步：RAGAS 多维度评测（{len(ragas_dataset)} 条数据）...")
    results = run_evaluation(ragas_dataset)

    # ── 第四步：输出报告 ──
    # RAGAS 返回的结果可能是一个对象，需要转为 dict
    scores = {}
    metric_keys = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    for key in metric_keys:
        try:
            score = results[key]
            if hasattr(score, "item"):
                score = float(score.item())
            scores[key] = float(score)
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"无法获取 {key} 分数: {e}")
            scores[key] = 0.0

    # 保存 raw 结果
    results_file = dataset_dir / "eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    logger.info(f"评测结果已保存: {results_file}")

    # 打印报告
    print_report(scores)


if __name__ == "__main__":
    asyncio.run(main())
