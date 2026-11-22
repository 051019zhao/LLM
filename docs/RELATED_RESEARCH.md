# 模型来源识别相关论文

这道题要求区分 18 个具体模型。用于比较论文结果时，先确认论文预测的是模型版本、模型家族，还是人类与模型两类，再看数据划分和指标。

| 论文 | 作者及出版状态 | 可借鉴的方法与适用范围 |
|---|---|---|
| [Idiosyncrasies in Large Language Models](https://proceedings.mlr.press/v267/sun25z.html) · [GitHub](https://github.com/locuslab/llm-idiosyncrasies) | Mingjie Sun、Yida Yin、Zhiqiu Xu、J. Zico Kolter、Zhuang Liu，ICML 2025 | 在相同提示词下采集多个模型的答案，再微调文本嵌入。论文报告五种聊天模型的准确率为 97.1%，而 Qwen 2.5 家族内部四种规模为 59.8%。这两项都是论文数据上的准确率，与比赛的 18 类 Balanced Accuracy 不直接可比。 |
| [Who's Behind the Text? Robust Model Authorship Attribution](https://arxiv.org/abs/2610.19583) · [GitHub](https://github.com/komseok0109/ATLAS) | Minseok Ko、Hyunseok Lee、Woomin Song、Beomjun Kim、Jinwoo Shin、Kunwoong Kim，2026 年 arXiv 预印本 | ATLAS 在冻结模型上训练家族条件适配器，比较文本在各家族条件下的似然。评估了领域及模型版本变化时的家族归因。可借鉴其似然特征，再为同家族的具体版本训练分类器。 |
| [Few-Shot Detection of Machine-Generated Text using Style Representations](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7e54a2592e3f51b9bed625a780310424-Abstract-Conference.html) · [GitHub](https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024) | Rafael Rivera Soto、Kailin Koch、Aleem Khan、Barry Chen、Marcus Bishop、Nicholas Andrews，ICLR 2024 | 用作者风格嵌入做少样本机器文本检测。可将风格向量接到目前的 18 类分类器上做对照。论文检测任务的结果不能替代本题的版本级验证。 |
| [Authorship Attribution Using Text Distortion](https://aclanthology.org/E17-1107/) | Efstathios Stamatatos，EACL 2017 | 在跨主题作者归因中遮蔽主题相关文本，保留结构线索。第三轮把此思路迁移到模型归因，并单独测试用训练集提示词估计和扣除共享主题方向。后者为本项目的实现。 |

先测试 CPU 可运行的主题遮蔽和核岭分类。更重的预训练风格编码器可单独比较，重点看模型家族内部的召回率，以及在新提示词上的表现。第三轮实验只读取官方训练数据。
