# 消融实验对比报告

本报告用于对比加入题目难度感知模块的 DA-AKT 与关闭难度模块的普通 AKT。

| run       | use_difficulty | best_valid_auc | test_auc | test_acc | test_loss | trainable_parameters |
| --------- | -------------- | -------------- | -------- | -------- | --------- | -------------------- |
| DA-AKT    | True           | 0.5962         | 0.6337   | 0.6015   | 0.6759    | 103806               |
| Plain-AKT | False          | 0.5410         | 0.5478   | 0.5330   | 0.7116    | 94846                |

- Test AUC 差值（DA-AKT - Plain-AKT）：0.0860
- Test ACC 差值（DA-AKT - Plain-AKT）：0.0685

说明：压缩包中的结果来自快速样例数据，主要用于证明工程流程可运行；正式课程报告可替换为 ASSIST2009 等真实数据集后的结果。
