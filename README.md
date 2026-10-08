# MagWorld Virtual Cell（3dworld）

基于 AI 的虚拟细胞研究代码库：用单细胞组学 + 电镜断层数据构建可预测、可生成的「虚拟细胞」。包含两条研究线：

- **World Model 线**：从单细胞转录组学习细胞状态流形与扰动动力学，对未见基因/药物组合做零样本 in-silico 预测。
- **3D 细胞器生成线（subflow3d）**：从 2D 电镜断层分割数据生成带细胞器的 3D 细胞结构。

---

## 一、World Model 线（`worldmodel/` + `pipeline/` + `launchers/`）

### 训练数据

**单细胞 RNA-seq 数据集**（`pipeline/01_core_pipeline.py`，统一 QC → 双细胞去除 → 归一化 → HVG → PCA → 聚类 → UMAP 预处理）：

| 数据集 | 用途 |
|---|---|
| pbmc3k | 基础流程验证 |
| Datlinger 2017（CRISPRi，33 靶基因） | 基因级留一扰动预测基准 |
| Aissa 2021（药物组合） | 跨数据集双药联合效应预测 |
| Norman 2021 | 组合扰动基准（`pipeline/09`） |
| Replogle 2022（RPE1 大规模 KO） | 大规模扰动基准（`pipeline/11-12`） |
| pancreas endocrinogenesis | 动力学/发育轨迹建模（`pipeline/17`） |
| dentategyrus / xiehon | 附加验证 |

**先验知识库**（`worldmodel/build_*.py` 构建为图/特征先验，融入模型 v14/v18/v21/v24+）：

- DGIDB（药物-基因互作，30 万+ 互作）
- TRRUST（TF-靶基因调控）
- STRING / Reactome（蛋白互作/通路）
- 共表达谱（对照细胞基因 PCA 嵌入）
- HGNC 基因家族特征

### 算法演进（`worldmodel/model_world_h1_v*.py`，13 → 37）

| 版本 | 核心思想 |
|---|---|
| v13/v14 | World VAE（lat=32）学习对照细胞流形 + 线性扰动算子 W（岭回归）：`Δ(gene) = W @ E[gene]`，对未见靶点零样本预测 |
| v16-v18 | 图先验融合（GAT / 多源先验加权） |
| v20-v21 | 药物侧信息与家族特征融合 |
| v24-v25 | **磁流（curie-weiss 风格场）模型**，`model_world_h1_v25.py` 为冠军底座 |
| v30-v34 | 规模与训练策略迭代（辅助任务、DGIDB 图先验、正则化） |
| v35-v36 | 局部校验与结构精简（`test_v35_37_local.py`） |
| **v37（当前最新）** | v25 磁流冠军 + **kq 作用力因果链**：扰动因子场向量 k 与基因查询轴 q 逐元素相乘得向量力 E，调制神经值向量 v，经基因响应轴软归一化投影出单基因上下调效应（负=下调，正=上调），无 kNN、单次前向 |

动力学底座（`pipeline/17_worldmodel_dynamics.py`）：VAE latent 作为世界状态 z → 编码器局部线性化估计 latent 速度 → MLP 学习转移场 fθ: z → dz/dt → in-silico 干预 rollout。

### 评估基准

- `pipeline/09` Norman 组合扰动；`pipeline/12` 跨数据集迁移；`pipeline/19-21` GEARS 对比与迁移、JEPA 对比
- 指标：delta_pearson / delta_mae_norm / DEG top-50 / 判别排名；动力学指标：velocity MSE/cosine、未来状态预测
- in-silico KO（`pipeline/07`）、去噪（`pipeline/14`）、单细胞级推理（`pipeline/13`）

---

## 二、3D 细胞器生成线（`subflow3d/`）

### 训练数据

**EMPIAR-10791，1857 号断层「Obese Climp-63」**（FIB-SEM 聚焦离子束扫描电镜）：

- 六类分割图已下载、校验、解压：**每类 3629 张，9650×9700 像素，uint8，共 1.6TB**
- 细胞分割为实例标签（0–57 个细胞）；线粒体等五类细胞器为 {0,255} 二值掩码
- 存储：内部集群数据盘 `/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked/`

**数据集构建**（`build_empiar_ds3.py`，产出 `empiar1857-mito-p2-v3.npz`，136 个样本）：

1. 线粒体感知选窗：9 个 z 窗口（300–3350），每窗 64 层投影降采样到 256² 网格打分
2. 贪心取 top-4 不重叠窗口，每窗切 4 个 64³ 子块（选窗坐标 clamp 到 9650/9700-256 防边缘不齐）
3. 过滤条件：细胞占比 >5% 且线粒体占比 >0.5%；最终样本线粒体占比跨度 0.8%–92%

### 算法：subflow 流匹配

核心脚本 `train_volume.py`（Bounded 3D conditional Flow Matching）：

- **目标**：生成 64³ 线粒体 3D 掩码体积
- **条件**（双通道）：细胞轮廓掩码 + 细胞 SDF（符号距离场，`scipy distance_transform_edt`）
- **路径**：最优传输（OT）路径；**采样**：欧拉法 50 步
- **架构**：3D 全卷积速度场网络 `VolumeVelocityField`，冠军配置 32 通道 / 4 残差块（10k 步，bs=8）
- 损失：focal 加权版（`train_focal.py`，w = 1+9·target，批内归一化）与 plain BCE 版并存
- 级联采样（`cascade_sample.py`）：P0/P1/P2 三级流匹配以留出测试细胞为条件级联生成；`render3d.py` / `render_gif.py` 做 3D 渲染

### 实验结果（2026-10-01，同批 8 测试样本，阈值按 GT 前景比例对齐）

| 实验 | IoU | DICE | 结论 |
|---|---|---|---|
| v3 掩码目标，32ch/4块，10k 步 | **0.214** | 0.331 | 当前最优 |
| v4 SDF 回归目标 | 0.074 | 0.133 | 证伪：密度失准，零面级细结构被洗掉 |
| focal 加权损失 | 0.212 | 0.327 | 持平：类别不平衡不是瓶颈 |

**瓶颈判断**：5 层 conv 有效感受野限制长程形态建模（线粒体网络状分支），下一步方向为跳跃连接 + 多尺度条件 3D U-Net。

---

## 目录结构

```
worldmodel/    扰动 world model：模型 v13→v37、训练/评估/先验构建/调试脚本
pipeline/      编号 00-21 全流程（质控→速度场→通路→CCC→基准→KO→GEARS/JEPA 对比）
subflow3d/     3D 线粒体生成：数据构建、流匹配训练、focal/SDF 消融、级联采样、渲染
launchers/     集群提交/训练脚本（VCC_TOKEN 用环境变量注入，勿硬编码）
```

## 复现与环境

- Python + PyTorch（集群 GPU 训练），scanpy/anndata（组学线），tifffile/scikit-image（电镜线）
- 数据不随仓库分发：1.6TB 分割图在内部集群数据盘；组学数据从公开来源下载（`pipeline/00_download_data.py`）
- 实验管理：提交脚本见 `launchers/submit_*.sh`，token 一律 `export VCC_TOKEN` 注入

## 安全说明

- 含内部主机名/凭证的 SSH 辅助脚本未入库
- 历史提交曾包含的 PAT 已清除并 force-push 覆盖；**该 PAT 应视为已泄露，请轮换**
