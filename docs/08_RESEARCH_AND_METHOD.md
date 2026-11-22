# 08：为什么更换数据路线，以及如何提交

目标是争取本队 Kaggle Balanced Accuracy 超过 0.41。本文中的本地验证、他人公开报告和本队真实公榜是三种证据，不互相替代。

## 从现有结果得到的判断

本队 03、04、06 的真实公榜分别为 0.368、0.363、0.361。07 对字符、排版与 NB-SVM 的受控改进较小，提升区间包含零。官方仅有 150 个提示词、每类约 150 段文本；在同一小训练集上继续增加特征，容易继续提高训练拟合而难以提高新提示词上的效果。

本轮先用相同提示词划分比较数据来源，再决定是否引入昂贵编码器。以下是实际阅读的公开资料；只借鉴方法，未直接复制这些仓库的实现代码。

| 公开来源 | 可借鉴的证据 | 本轮取舍 |
|---|---|---|
| [SSanjay0614 的项目](https://github.com/SSanjay0614/Name-That-LLM-Kaggle)及[公开 SVC 数据消融](https://github.com/SSanjay0614/Name-That-LLM-Kaggle/blob/main/kernel_v16/ntl_v16_grouped_svc_source_ablation.ipynb) | 公开训练池与多种文本视图值得独立评估；作者明确区分组件验证和整套融合的实赛成绩 | 实现 CPU 多视图 SVM；不引入其依赖很多权重的数据管线 |
| [stewari1210 的项目](https://github.com/stewari1210/name-that-llm) | 冻结语言模型统计与家族专家可补充词频信号；作者记录了生成式 LoRA 的负结果和按提示词验证 | 保留为后续 GPU 分支；历史 05 的家族拆分未稳定优于 04，不能直接认为层次分类必然更强 |
| [richardharnisch 的项目](https://github.com/richardharnisch/name-that-llm) | 比较了小样本和公开大数据条件；参数或正则化变化未自动解决小样本泛化 | 先检验数据规模和来源，比继续堆模型更直接 |
| [Open Pool 数据卡](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/18) | 记录来源、模型路由风险、重复与训练提示词排除；也披露数据与官方文本的差异 | 固定 v18；不用依赖完整官方训练标签得到的 match_weight/style_score；记录其既有清洗局限 |

这些来源的模型成绩均为作者报告，不能计入本队提交次数或课程成绩。论文背景另见 [RELATED_RESEARCH.md](RELATED_RESEARCH.md)，其中不同标签粒度和数据上的论文分数不能与本题直接比较。

## 08 的完整方法

1. 读取官方训练数据和固定的 Open Pool v18，核对外部 parquet 的 SHA-256。
2. 外部数据排除训练提示词再生成标识、Gemini 3.7 路由风险标识、空值、非法标签、规范化后的重复和标签冲突，以及多段落、Markdown 包装和 40–250 词范围以外的记录。官方文本的精确重复也排除。
3. 每类用固定种子 1908 均匀抽取 3,000 条，共 54,000 条。这个均衡针对训练采样，**不对测试预测数量作均衡配额**。
4. 同一训练折上拟合四组 TF-IDF：保留大小写的字符 3–5 元组、词/标点 1–2 元组、保留长度/大小写的词形字符元组、保留功能词的结构元组。权重 1、0.7、0.5、0.5。
5. 拟合 LinearSVC(C=0.5)。官方样本基础权重为 12，外部样本为 1。先计算每类的有效权重总量，再作类别平衡，最后使样本权重平均为 1。这样不会把来源数量差直接当作测试先验。
6. 每条测试文本独立预测，不使用隐藏提示词标签、不假设每个提示词有完整 18 个模型、不使用 Hungarian 硬分配，也不把预测按类别数量强制拉齐。

验证词表只看该折训练部分与固定外部池，不看验证文本。保留本轮 30 个提示词，只在候选冻结后评估。需要披露：旧实验曾用过全部 150 个提示词，外部数据发布者的排版清洗也参考过完整官方训练集；因此本轮留出不是历史上完全未使用的独立测试集，区间也未计入这类适应性偏差。

训练准确率可以达到 100%，仍有较大的训练/验证差距；本轮改善泛化的证据是相同提示词划分下的提升，不能声称已经消除过拟合。

## 需要的输入与运行

### Kaggle

导入 [08_external_data_submission.ipynb](../notebooks/08_external_data_submission.ipynb)。右侧添加比赛 **Name That LLM** 和 [**Name That LLM Open Pool，版本 18**](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/versions/18)。旧版 03 只需比赛输入，08 还必须有这份外部数据。文件哈希不符会明确停止，不会悄悄降级为旧模型。

选择 CPU，关闭 Internet，Save Version → Save & Run All。等保存版本成功，并确认 Output 中有 `submission.parquet`，再点比赛页 Submit Prediction，选这次保存的版本。互动预览只有 3 行是正常的，正式评分重新读取约 4,000 条隐藏数据。

### Windows 本地

从仓库目录运行：

```powershell
python -m pip install -r requirements.txt
python experiments/download_public_data.py
python run_local.py --round 8 --output-dir local_results/round08_submission
```

如果网络下载中断，重跑下载命令可续传。训练文件仍自行从已加入的比赛下载到 `data/`，不要上传到 GitHub。

完整复现固定候选的开发三折和本轮留出评估时，在最后一行加 `--validate`；不会根据验证分数自动搜索参数。数据来源对照使用 `experiments/evaluate_external08.py --kind official|baseline03|extended|instyle|open`，另两份扩展集可通过下载脚本的 `--source extended` 或 `--source instyle` 获取。各选项需要分别执行，不把竖线当作命令参数。

逐折结果、候选冻结依据、置信区间和运行检查见 [round08 实验记录](../experiments/round08/README.md)。新版本的公榜得分须等本队正式提交后登记。

## 外部数据许可和归属

Open Pool 由 crtproduction 整合，v18 原始来源及许可逐项列在 [EXTERNAL_DATA_ATTRIBUTION.csv](EXTERNAL_DATA_ATTRIBUTION.csv)，完整出处以数据集的 `sources.csv`、`source_protocol.csv` 和数据卡为准。合并数据采用 CC BY-SA 4.0；上游包含 CC BY 4.0、CC BY-SA 4.0、CC0。这里提供下载链接、版本、哈希和训练代码，未把原始外部文本或比赛数据重新发布到仓库。

另外测试的两份数据由 prvsiyan 发布：[Extended v13](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/versions/13)，CC BY 4.0；[in-style pool v2](https://www.kaggle.com/datasets/prvsiyan/ntl-public-in-style-pool-v1/versions/2)，CC BY-SA 4.0，底层作者归属见其数据卡。这两份用于对照，08 正式路径只挂载 Open Pool v18。
