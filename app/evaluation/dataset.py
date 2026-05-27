"""评测数据集模块

从知识库文档自动生成 QA 对，用于 RAGAS 评测
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict

from datasets import Dataset
from loguru import logger

from app.core.llm_factory import llm_factory
from app.services.rag_agent_service import rag_agent_service
from app.tools.knowledge_tool import retrieve_knowledge, current_user_id, latest_source_files


# ── 数据集定义 ──

@dataclass
class QAPair:
    question: str
    ground_truth: str
    context: str = ""           # 用于生成标准答案的参考文档
    source_file: str = ""


# ── 自动生成 QA 对 ──

QA_GENERATION_PROMPT = """你是一个评测数据集生成专家。根据以下文档内容，生成 5 个高质量的问答对。

要求：
1. 问题必须基于文档内容，不能编造文档中没有的信息
2. 问题类型要多样化：有事实类、有推理类、有排查步骤类
3. 答案要准确、完整，引用文档中的具体内容
4. 答案控制在 100-300 字之间
5. 问题语言：中文

输出格式（严格 JSON 数组）：
[
  {{
    "question": "问题1",
    "answer": "答案1"
  }},
  {{
    "question": "问题2",
    "answer": "答案2"
  }}
]

文档标题：{title}
文档内容：
{content}

请生成 5 个问答对："""


def generate_qa_pairs_from_docs(docs_dir: str = "aiops-docs", output_path: str = "app/evaluation/datasets") -> list[QAPair]:
    """
    从 aiops-docs 的每篇文档用 LLM 自动生成评测 QA 对

    Args:
        docs_dir: 知识库文档目录
        output_path: 输出目录

    Returns:
        list[QAPair]: 所有生成的 QA 对
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"文档目录不存在: {docs_dir}")
        return []

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = llm_factory.create_chat_model(
        model="qwen-max",
        temperature=0.7,  # 稍高温度增加多样性
        streaming=False,
    )

    all_qa_pairs = []

    md_files = list(docs_path.glob("*.md"))
    logger.info(f"从 {len(md_files)} 篇文档生成评测数据集...")

    for md_file in md_files:
        title = md_file.stem
        content = md_file.read_text(encoding="utf-8")

        # 截取前 3000 字符（避免超出 token 限制）
        content_truncated = content[:3000]

        prompt = QA_GENERATION_PROMPT.format(title=title, content=content_truncated)

        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # 提取 JSON 数组
            # 处理 LLM 可能包裹在 ```json ``` 中的情况
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if json_match:
                response_text = json_match.group(1)

            qa_list = json.loads(response_text)

            for qa in qa_list:
                pair = QAPair(
                    question=qa["question"],
                    ground_truth=qa["answer"],
                    context=content_truncated,
                    source_file=md_file.name,
                )
                all_qa_pairs.append(pair)

            logger.info(f"  {title}: 生成 {len(qa_list)} 个 QA 对")

        except Exception as e:
            logger.error(f"  {title}: 生成失败 - {e}")
            continue

    # 保存到文件
    dataset_file = output_dir / "qa_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(p) for p in all_qa_pairs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"评测数据集已保存: {dataset_file} ({len(all_qa_pairs)} 个 QA 对)")

    # 也存一份纯问题列表，方便手动 review
    questions_file = output_dir / "qa_questions.txt"
    with open(questions_file, "w", encoding="utf-8") as f:
        for i, pair in enumerate(all_qa_pairs, 1):
            f.write(f"Q{i}: {pair.question}\n")
            f.write(f"A{i}: {pair.ground_truth[:200]}...\n")
            f.write(f"来源: {pair.source_file}\n\n")

    return all_qa_pairs


# ── 将 QA 对转换为 RAGAS Dataset ──

def build_ragas_dataset(qa_pairs: list[QAPair], session_id: str = "eval_session") -> Dataset:
    """
    用真实的 RAG 检索 + 生成流程跑一遍，构建 RAGAS 评测数据集

    对每个问题：
    1. 调用 retrieve_knowledge 获取 contexts（检索结果）
    2. 调用 rag_agent_service.query 获取 answer（RAG 回答）
    3. 与 ground_truth（标准答案）组合

    Args:
        qa_pairs: QA 对列表
        session_id: 评测专用 session ID

    Returns:
        Dataset: HuggingFace Dataset（RAGAS 格式）
    """
    eval_data = []
    token = current_user_id.set("eval_user")

    try:
        for i, pair in enumerate(qa_pairs):
            logger.info(f"处理 [{i+1}/{len(qa_pairs)}]: {pair.question[:50]}...")

            try:
                # 1. 检索上下文
                context_text, raw_docs = retrieve_knowledge.invoke({"query": pair.question})
                contexts = [doc.page_content for doc in raw_docs] if raw_docs else [context_text]

                # 2. 生成回答
                answer = ""
                try:
                    answer = rag_agent_service.query(
                        question=pair.question,
                        session_id=session_id,
                        user_id="eval_user",
                    )
                except Exception as e:
                    logger.warning(f"  回答生成失败: {e}")
                    answer = f"[生成失败] {e}"

                eval_data.append({
                    "question": pair.question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": pair.ground_truth,
                })

                logger.info(f"  生成完成: answer={len(answer)}字, contexts={len(contexts)}条")

            except Exception as e:
                logger.error(f"  处理失败: {e}")
                continue

    finally:
        current_user_id.reset(token)

    # 转换为 HuggingFace Dataset
    if not eval_data:
        logger.error("未生成任何评测数据")
        return Dataset.from_dict({
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        })

    dataset = Dataset.from_dict({
        "question": [d["question"] for d in eval_data],
        "answer": [d["answer"] for d in eval_data],
        "contexts": [d["contexts"] for d in eval_data],
        "ground_truth": [d["ground_truth"] for d in eval_data],
    })

    logger.info(f"RAGAS Dataset 构建完成: {len(dataset)} 条")
    return dataset
