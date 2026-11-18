# Name That LLM｜前两次 Kaggle 提交评审

**评审日期：2026-11-18**  
**当前公榜基线：03，Balanced Accuracy = 0.368**

## 提交记录

| 本地代码 | Kaggle 运行版本 | 方法 | 状态 | 公榜分数 |
|---|---|---|---|---:|
| `03_optimized_submission.ipynb` | [notebook07e3726630 · Version 4](https://www.kaggle.com/code/veltsofuad/notebook07e3726630?scriptVersionId=371029596) | 字符与风格特征，线性 SVM + Extra Trees | Succeeded | **0.368** |
| `04_topic_robust_submission.ipynb` | [notebook4bce5f502c · dan03](https://www.kaggle.com/code/veltsofuad/notebook4bce5f502c?scriptVersionId=371044280) | 在 03 上加入主题遮蔽核岭分支 | Succeeded | 0.363 |

表中的 03、04 是本地文件编号，Version 4、dan03 是 Kaggle 保存的运行版本。03 的源文件已同步至 GitHub，上传后核对了文件哈希。[03 源代码](https://github.com/051019zhao/LLM/blob/d87b72a802985bc028827a131bf265d73a1ccaab/notebooks/03_optimized_submission.ipynb)

## 两版是怎么做的

**03 用字符和写作风格判断模型来源。** 线性 SVM 读取字符 TF-IDF、字符形态和风格统计。Extra Trees 使用句长、功能词、标点和排版等风格统计。预测时将归一化的 SVM 分数与树模型概率按 **75:25** 融合。提交配置为 `blend_c1_w25`。

**04 加了一条弱化主题词的分支。** 保留常用词、标点和大小写模式，把其他词编码为长度标记，再用遮蔽文本的字符 TF-IDF 与已有特征构造相似度核，训练核岭分类器。最终按 **55% 的 03 分数 + 45% 的核岭分数** 融合。提交配置为 `blend_masked_kernel`。[主题遮蔽的论文参考](https://aclanthology.org/E17-1107/)

## 本地验证与公榜为什么分开看

官方训练集有 **2,699 条文本、150 个提示词和 18 个模型类别**。本地验证按 `prompt_id` 分组，同一提示词的所有回答都放在同一折。特征词表和缩放参数只用对应训练折拟合。Balanced Accuracy 对 18 个类别的召回率取平均，每类权重相同。[官方数据](https://www.kaggle.com/competitions/name-that-llm/data) · [评分方式](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)

| 同一组验证样本 | 03 | 04 | 04 相对变化 |
|---|---:|---:|---:|
| 40 个提示词留出集 | 37.10% | 38.21% | +1.11 个百分点 |
| 后续五折分组复核均值 | 36.63% | 37.45% | +0.81 个百分点 |
| 五折复核的折间标准差 | 0.81 个百分点 | 1.93 个百分点 | +1.12 个百分点 |
| Kaggle 公榜 | 36.8% | 36.3% | −0.5 个百分点 |

本地差值按未舍入的分数计算。

04 的本地提升没有在公榜复现。按提示词对五折预测做配对重采样，04 相对 03 的 95% 区间约为 **−0.22 至 +1.85 个百分点**，跨过零。变化也分文本长度：80 词以内只提升 **0.23 个百分点**，81–150 词提升 **1.36 个百分点**。这些证据说明融合效果会随样本变化，但公榜只给总分，无法定位隐藏测试集上具体错分的文本或类别。赛事公榜使用约 40% 的测试数据，最终私榜取其余部分。[Kaggle 排行榜说明](https://www.kaggle.com/competitions/name-that-llm/leaderboard)

## 队友接手

1. **以 03 的 0.368 作为当前公榜基线，保留 04 作为有效对照。** 后续实验选择固定的提示词分组，同时记录均分、折间波动和每类召回率。
2. **每次提交记录模型版本、唯一主要改动、本地验证分和公榜分。** 核岭分支降到 15% 权重的候选在同组五折仅得 **36.82%**，暂不单独提交。
3. **复现 Kaggle 运行。** 打开对应的保存版本，在有权限的 Kaggle 账号中复制或导入 Notebook，挂载 **Name That LLM** 比赛数据，用 CPU 并将 **Internet** 设为 **Off**。运行完成后确认输出是 `/kaggle/working/submission.parquet`，包含 `id`、`model` 两列，再提交运行成功的版本。互动运行看到 3 行样本属于正常现象，比赛提交时会使用隐藏测试集重跑。[官方提交流程](https://www.kaggle.com/docs/competitions)
