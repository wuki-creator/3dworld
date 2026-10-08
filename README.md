# MagWorld Virtual Cell

基于 AI 的虚拟细胞研究代码库。包含两条研究线：

## 1. World Model 线（`worldmodel/` + `pipeline/` + `launchers/`）

单细胞扰动响应预测（virtual cell world model）：

- 图谱构建 → 速度场 → 通路/TF/GSEA → 细胞通讯（CCC）全流程（`pipeline/00-21` 编号脚本）
- 模型迭代 v13 → v37（`worldmodel/model_world_h1_v*.py`）：图先验融合、GEARS 迁移、JEPA、动力学世界模型
- 在 Norman / Replogle 等扰动基准上评估（`pipeline/09-13`）
- in-silico KO、去噪、单细胞级推理（`pipeline/07,13,14`）

## 2. 3D 细胞器生成线（`subflow3d/`）

从 2D 电镜断层数据生成带细胞器的 3D 细胞结构（EMPIAR-10791，1857 Obese Climp-63，FIB-SEM）：

- 方法：subflow 流匹配，条件 = [细胞轮廓, SDF]，目标 = 线粒体 3D 掩码，64³ crops，OT 路径 + 欧拉采样 50 步
- 数据集构建：线粒体感知选窗（`build_empiar_ds*.py`）
- 级联采样 + 3D 渲染（`cascade_sample.py`, `render3d.py`）

### 关键结果（2026-10-01）

| 实验 | IoU | DICE | 结论 |
|---|---|---|---|
| v3 掩码目标，32ch/4块，10k 步 | **0.214** | 0.331 | 当前最优 |
| v4 SDF 回归目标 | 0.074 | 0.133 | 证伪：密度失准，细结构消失 |
| focal 加权损失 | 0.212 | 0.327 | 持平：类别不平衡不是瓶颈 |

瓶颈判断：5 层 conv 有效感受野限制长程形态建模，下一步方向为跳跃连接 + 多尺度条件 3D U-Net。

## 说明

- 数据（1.6TB 分割图、.npy/.npz 中间产物）不随仓库分发，位于内部计算集群数据盘。
- 含内部主机名的 SSH 辅助脚本未入库。
- 历史调试/检查脚本保留原样，作为实验过程记录。
