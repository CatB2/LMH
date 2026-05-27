"""RAGAS 评测指标模块

每个指标的含义、计算方式、面试话术
"""

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.core.llm_factory import llm_factory
from app.services.vector_embedding_service import vector_embedding_service
from loguru import logger

# ── RAGAS 评测指标说明（面试 / 简历用） ──

METRIC_EXPLANATIONS = {
    "faithfulness": {
        "name_cn": "忠实度",
        "range": "0.0 ~ 1.0，越高越好",
        "what": "回答中的每一条断言，是否都能在检索到的文档中找到依据。",
        "why_matters": "衡量模型是否'编造'内容。RAG 系统最核心的指标——低忠实度说明模型在胡说八道。",
        "how": "RAGAS 将回答拆解为原子断言，逐条与上下文对比，计算有依据的比例。",
        "interview_script": (
            "忠实度衡量答案是否基于知识库。我通过 RAGAS 框架将模型回答拆解为原子事实，"
            "逐条验证是否能在检索到的上下文中找到支撑，避免大模型幻觉。"
            "这个指标直接决定了 RAG 系统是否可用——如果忠实度低于 0.7，用户将无法信任系统。"
        ),
    },
    "answer_relevancy": {
        "name_cn": "答案相关性",
        "range": "0.0 ~ 1.0，越高越好",
        "what": "回答是否直接、完整地回应了用户的问题，有没有答非所问。",
        "why_matters": "用户问 A 你答 B，检索再准也没用。反映 LLM 理解意图和聚焦的能力。",
        "how": "RAGAS 用 LLM 从回答反向生成问题，计算反向问题与原问题的语义相似度。",
        "interview_script": (
            "答案相关性通过反向生成技术衡量——让 LLM 根据答案'猜'原始问题是什么，"
            "再计算猜测的问题与真实问题的语义相似度。如果答案牛头不对马嘴，反向生成的问题会和原问题差距很大。"
            "这个指标帮助我们发现 prompt 设计或检索结果段落选择中的问题。"
        ),
    },
    "context_precision": {
        "name_cn": "上下文精度",
        "range": "0.0 ~ 1.0，越高越好",
        "what": "检索到的文档中，有多少是真正相关的。无关文档排在前面会拉低分数。",
        "why_matters": "衡量检索系统的信噪比。精度低意味着大量噪声文档混入，浪费 token 预算且干扰 LLM。",
        "how": "RAGAS 判断每个检索到的文档是否与标准答案相关，按排名加权（靠前的更重要）。",
        "interview_script": (
            "上下文精度衡量检索质量——返回的文档里有多少是真正有用的。"
            "我们采用 RRF 融合算法（向量检索 + BM25）就是为了提升这个指标。"
            "向量检索擅长语义匹配，BM25 擅长关键词精确命中，两者互补能显著提高精度。"
        ),
    },
    "context_recall": {
        "name_cn": "上下文召回率",
        "range": "0.0 ~ 1.0，越高越好",
        "what": "标准答案所需的信息，检索结果覆盖了多少。漏掉的越多，召回率越低。",
        "why_matters": "衡量知识是否'搜全了'。召回低说明知识库有信息但检索没捞到，答案会不完整。",
        "how": "RAGAS 用 LLM 将标准答案拆成句子，逐句判断是否能在检索文档中找到。",
        "interview_script": (
            "上下文召回率衡量检索的覆盖度——知识库里有的东西，检索系统能不能'搜全'。"
            "高召回对运维场景特别重要：漏掉一个关键日志或告警信息，可能导致故障定位失败。"
            "我们通过调整 top_k 和 RRF 融合权重来平衡精度和召回率。"
        ),
    },
    "answer_correctness": {
        "name_cn": "答案正确性",
        "range": "0.0 ~ 1.0，越高越好",
        "what": "答案在事实层面是否与标准答案一致。综合了语义相似度和事实准确性。",
        "why_matters": "最终用户关心的就是这个——你说的对吗？这是所有其他指标的目标。",
        "how": "RAGAS 综合计算：1) 回答与标准答案的语义相似度 2) 事实层面的 TP/FP/FN。",
        "interview_script": (
            "答案正确性是最终指标——它综合了语义相似度和事实准确性两个维度。"
            "我们用 F1 的思想来评估：真正（答对的）、假正（编造的）、假负（遗漏的）。"
            "正确性高说明检索精度、忠实度、相关性等多个环节配合良好。"
        ),
    },
}


def _build_llm():
    """构建 RAGAS 评测用的 LLM（用项目现有的模型工厂）"""
    model = llm_factory.create_chat_model(
        model="qwen-max",
        temperature=0.0,  # 评测用确定性输出
        streaming=False,
    )
    return LangchainLLMWrapper(model)


def _build_embeddings():
    """构建 RAGAS 评测用的 Embedding"""
    return LangchainEmbeddingsWrapper(vector_embedding_service.embeddings)


def run_evaluation(test_dataset: Dataset) -> dict:
    """
    运行 RAGAS 多维度评测

    Args:
        test_dataset: HuggingFace Dataset，需包含字段：
            - question: 用户问题
            - answer: RAG 系统生成的回答
            - contexts: 检索到的文档列表
            - ground_truth: 标准答案

    Returns:
        dict: 评测结果，包含各指标分数和解释
    """
    logger.info("开始 RAGAS 评测...")

    eval_llm = _build_llm()
    eval_embeddings = _build_embeddings()

    run_config = RunConfig(
        max_workers=1,          # 串行调用避免 API 限流
        timeout=120,            # 单个样本超时
    )

    result = evaluate(
        dataset=test_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )

    logger.info("RAGAS 评测完成")
    return result


def print_report(scores: dict):
    """打印评测报告，包含指标含义解释"""
    print()
    print("=" * 70)
    print("  SuperBizAgent RAG 系统评测报告")
    print("=" * 70)
    print()

    # 指标表格
    print(f"  {'指标':<16} {'分数':>6}  {'评价':<20}")
    print(f"  {'─' * 16:<16} {'─' * 6:>6}  {'─' * 20:<20}")

    thresholds = {
        "faithfulness": (0.80, 0.60),
        "answer_relevancy": (0.80, 0.60),
        "context_precision": (0.75, 0.55),
        "context_recall": (0.75, 0.55),
        "answer_correctness": (0.75, 0.55),
    }

    for key, score in scores.items():
        explanation = METRIC_EXPLANATIONS.get(key, {})
        name = explanation.get("name_cn", key)
        good, bad = thresholds.get(key, (0.80, 0.60))
        if score >= good:
            grade = "🟢 优秀"
        elif score >= bad:
            grade = "🟡 一般"
        else:
            grade = "🔴 待优化"
        print(f"  {name:<16} {score:>6.4f}  {grade:<20}")

    print()
    print("─" * 70)
    print("  各指标详解（面试话术参考）")
    print("─" * 70)
    print()

    for key, info in METRIC_EXPLANATIONS.items():
        score = scores.get(key, "N/A")
        print(f"  【{info['name_cn']}】{info['range']}")
        print(f"  当前得分: {score}")
        print(f"  是什么: {info['what']}")
        print(f"  为什么重要: {info['why_matters']}")
        print(f"  怎么算的: {info['how']}")
        print(f"  面试话术: {info['interview_script']}")
        print()

    print("=" * 70)

    # ── 简历数据 ──
    print()
    print("  📋 可直接写入简历的数据：")
    print()
    for key, info in METRIC_EXPLANATIONS.items():
        score = scores.get(key, 0)
        print(f"  {info['name_cn']}: {score:.2%}")

    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"  综合均分: {avg:.2%}")
    print()
    print(f"  简历写法示例：")
    print(f"  \"基于 RAGAS 框架对多智能体 RAG 系统进行 5 维度评测，")
    print(f"   忠实度 {scores.get('faithfulness', 0):.1%}、答案相关性 {scores.get('answer_relevancy', 0):.1%}、")
    print(f"   上下文精度 {scores.get('context_precision', 0):.1%}、召回率 {scores.get('context_recall', 0):.1%}、")
    print(f"   答案正确性 {scores.get('answer_correctness', 0):.1%}，综合均分 {avg:.1%}\"")
    print()
