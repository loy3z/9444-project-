# COMP9444 Project 54 - Workflow

## 项目目标

使用深度学习对 EuroSAT 卫星图像进行土地覆盖分类，并比较 ResNet50、ViT 和 EfficientNet 的表现。

## 任务分工

| 负责人 | 任务 |
|---|---|
| 数据预处理 | 类别统计、分层划分、图像尺寸统一、归一化、数据增强；提供统一 DataLoader |
| ResNet50 | 完成 baseline；输出完整训练与评估结果 |
| ViT | 在相同数据划分和评估标准下训练与调参 |
| EfficientNet | 在相同数据划分和评估标准下训练与调参 |
| 总结报告 | 汇总实验设置、图表、模型比较、讨论与结论 |

## Workflow

1. **准备数据**
   - 统计每个类别的样本数量。
   - 使用 stratified split 固定划分 train / validation / test。
   - 固定 random seed，并共享相同的数据划分。
   - 确定统一的 resize、normalization 和基础 augmentation。

2. **训练 ResNet50 baseline**
   - 首先不使用 class weights。
   - 记录 loss、accuracy、macro-F1、per-class recall 和 confusion matrix。
   - 保存最佳模型、训练配置和结果。

3. **决定是否使用类别权重**
   - 综合类别数量和 ResNet50 baseline 的 per-class recall 判断。
   - 如果少数类别表现明显偏低，增加 class-weighted loss 实验。
   - 比较加权与不加权的 validation macro-F1，确定三种模型统一采用的方案。

4. **训练其他模型**
   - ViT 和 EfficientNet 使用同一数据划分、预处理和加权策略。
   - 三种模型使用统一评价指标。
   - 分别保存最佳模型和实验记录。

5. **比较与总结**
   - 比较 test accuracy、macro-F1、训练时间、模型大小和 confusion matrix。
   - 分析不同类别的错误，以及模型的优缺点。
   - 完成报告、代码整理和最终展示。

## 公平比较要求

- 三个模型必须使用同一份 train / validation / test 划分。
- test set 只用于最终评估，不能用于调参。
- 记录 random seed、超参数、最佳 epoch 和运行环境。
- 不要同时使用 `WeightedRandomSampler` 和 class-weighted loss，除非有明确实验依据。
