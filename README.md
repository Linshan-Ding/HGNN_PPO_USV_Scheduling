# 考虑能耗约束的多无人船协同任务调度（HGNN + PPO）

本项目研究多无人船（USV）协同任务调度问题：多艘无人船从原点基地出发，在航行能耗、作业能耗、电池容量与自动返航换电约束下完成一组作业时长为三角模糊数的测绘任务，优化目标为最小化最大完工时间（makespan）。

主方法为**双编码器异构二部图强化学习**：异构二部图状态表示 + HGNN 编码器 + Pairwise 联合评分策略 + 并行 rollout PPO 训练。对比方法包括 5 条调度规则和 4 个共享同一网络架构的 DRL 基线（A2C / DQN / DDQN / REINFORCE），另有 3 个消融变体（no_hgnn / shared_encoder / no_reward_norm）。

---

## 目录

1. [环境安装](#环境安装)
2. [数据集](#数据集)
3. [核心实验协议](#核心实验协议)
4. [完整实验工作流（一键步骤）](#完整实验工作流一键步骤)
5. [输出文件与论文表图对应关系](#输出文件与论文表图对应关系)
6. [冒烟验证与测试](#冒烟验证与测试)
7. [文件索引](#文件索引)

---

## 环境安装

```bash
pip install -r requirements.txt          # numpy pandas matplotlib torch visdom
```

训练默认开启 Visdom 实时可视化，请先在**单独终端**启动服务：

```bash
python -m visdom.server -port 8097
# 浏览器打开 http://localhost:8097 查看
```

若无需实时曲线，所有训练脚本均支持 `--no-visdom`。

## 数据集

- **公开 25 算例**：`data/public/u{N}_t{M}.csv`，N ∈ {2,4,6,8,10} × M ∈ {20,40,60,80,100} 全组合，`manifest.csv` 记录元数据。固定种子生成，电池容量 = 1.2 × 最大单任务往返能耗。重新生成：`python instance_generator.py`。
- **泛化算例（7 个更大规模）**：`data/scalability/`，包含 u12_t100（仅舰队规模控制组）与 {u10,u12,u15} × {t150,t200}，由 `scalability_experiment.py` 首次运行时自动生成（固定种子）。

## 核心实验协议

论文实验部分（表注逐字引用本节输出的 CSV 字段）遵循以下协议：

1. **轮询多算例训练（round-robin）**：每个学习方法**单次训练**（seed 0），总周期 K=5000，第 k 周期训练算例 `(k−1) mod 25`（manifest 顺序），每算例获得 200 次访问。
2. **每次访问必做确定性评估**：每周期结束后对当期算例做一次确定性贪婪 rollout 评估（固定步长评估会因 gcd 效应漏评算例）。
3. **逐算例最优 checkpoint**：评估刷新该算例历史最优时保存 `models/best_{instance_id}_seed{seed}.pth`（我方）或 `models/best_{ALG}_{instance_id}_seed{seed}.pth`（基线），共 25 个/方法，供泛化实验零样本加载。
4. **算例求解结果 = 最后 10 次访问**的评估 makespan 的 mean/std/min（无需额外加载模型重评估）。
5. **计时口径**：学习类方法报告平均每训练周期耗时（该算例全部访问的 `epoch_time_sec` 均值）；调度规则报告预热 + 计时的单次求解耗时；泛化大算例上所有方法报告 CPU 单线程零样本求解耗时；逐决策延迟 = 评估 episode 墙钟 / 决策数（ms）。
6. **统计检验**：25 对逐算例结果的单侧 Wilcoxon 符号秩检验（α=0.05）。

多算例训练的良定性来自**归一化奖励**：无折扣回报 ≈ −2·C_max/T̄，跨算例回报尺度一致（消融变体 no_reward_norm 正是验证这一点）。

## 完整实验工作流（一键步骤）

按顺序执行（①为常驻服务；②–⑤为训练，可在多机/多卡上并行；⑥–⑦为汇总与出图）：

```bash
# ① 启动 Visdom（单独终端，常驻）
python -m visdom.server -port 8097

# ② 我方方法：轮询训练 25 算例 × 5000 周期
#    （首次运行会自动计算并缓存 5 条规则的计时基线 results/rules_results.csv）
python multi_train.py

# ③ 三个消融变体（同一协议；full 变体即 ② 的结果，不重复跑）
python ablation_experiment.py
#    或单独跑某个变体: python multi_train.py --variant no_hgnn

# ④ 四个 DRL 基线（同一协议）
python -m drl_baselines.multi_run --algorithm A2C
python -m drl_baselines.multi_run --algorithm DQN
python -m drl_baselines.multi_run --algorithm DDQN
python -m drl_baselines.multi_run --algorithm REINFORCE

# ⑤ 泛化与可扩展性：全部方法从各自 u10_t100 最优 checkpoint 零样本评测 7 个大算例
python scalability_experiment.py

# ⑥ 从训练日志提取论文表格 CSV（last-10 统计、平均周期耗时、Wilcoxon）
python extract_results.py

# ⑦ 生成论文全部配图（PDF，直接放入论文仓库 figures/generated/）
#    每图一条命令，全部单行、默认路径齐全，Windows cmd/PowerShell 可直接粘贴：
python analyze_training_logs.py training_curves
python analyze_training_logs.py convergence_all25
python analyze_training_logs.py ablation_curves
python analyze_training_logs.py gap_heatmap
python analyze_training_logs.py decision_time_heatmap
python analyze_training_logs.py drl_gap_violin
python analyze_training_logs.py gap_by_tasks
python analyze_training_logs.py dumbbell
python analyze_training_logs.py gap_ecdf
python analyze_training_logs.py scalability
python analyze_training_logs.py gantt --gantt-instance u6_t60
#    或一次全部生成（gantt 在缺 checkpoint 时自动跳过）：
python analyze_training_logs.py all
#    IEEE 双栏版（T-ITS 论文仓库）：按 IEEEtran 栏宽 3.5 in / 页宽 7 in 重新排版所有图，输出 results/figures_ieee/
python analyze_training_logs.py all --layout ieee
```

各命令的输入默认值：曲线/小提琴/箱线类读 `results/training_logs/`；`gap_heatmap` 读 `results/main_results.csv`；`decision_time_heatmap` 读 `results/decision_time_grid.csv`；`scalability` 读 `results/main_results.csv` + `results/scalability_summary.csv`；`dumbbell` 读 `results/main_results.csv`；`gap_ecdf` 读 `results/drl_results.csv`；`gantt` 读 `data/public/` 实例与 `models/best_{instance}_seed0.pth`（网络规模参数需与训练一致：`--hidden-dim 256 --hgnn-layers 3 --n-heads 4`，即默认值）。输出统一为 `results/figures/*.pdf`（`--format png` 可出预览图）；`--layout ieee` 时输出 `results/figures_ieee/*.pdf`，字号与图幅按 IEEE 双栏版设定，直接放入 IEEE 论文仓库 `figures/generated/`。

训练期间 Visdom 展示 **25 条逐算例训练曲线**（`Eval Makespan by Instance` 与 `Gap vs Best Rule (%) by Instance` 两个多曲线窗口，每算例每 25 个周期新增一个数据点），以及损失/熵/耗时等汇总曲线。

常用参数（`multi_train.py` 与 `drl_baselines/multi_run.py` 同名）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-epochs` | 5000 | 总训练周期（= 25 算例 × 200 访问） |
| `--seed` | 0 | 随机种子 |
| `--hidden-dim / --hgnn-layers / --n-heads` | 256 / 3 / 4 | 网络规模 |
| `--n-trajectories` | 8 | 每周期采样轨迹数 |
| `--lr-decay-step` | 250 | LR ×0.95 的周期间隔（仅 multi_train） |
| `--instances` | 全部 25 | 逗号分隔算例子集（冒烟用） |
| `--no-visdom` | 关 | 关闭实时可视化 |
| `--epsilon-decay-epochs` | 2000 | DQN/DDQN 探索衰减（仅 multi_run） |

## 输出文件与论文表图对应关系

| 文件 | 生成脚本 | 论文对应 |
|---|---|---|
| `results/rules_results.csv` | multi_train（自动缓存） | 规则逐算例 makespan + 求解耗时 |
| `results/training_logs/PPO_full_public25_seed0_*.csv` | multi_train | 5.2 训练曲线数据源 |
| `results/training_logs/{ALG}_baseline_public25_seed0_*.csv` | multi_run | DRL 基线训练数据 |
| `results/training_logs/PPO_{variant}_public25_seed0_*.csv` | ablation_experiment | 消融训练数据 |
| `results/main_results.csv` | extract_results | **tab:main_results**（规则 R1–R5 vs 我方） |
| `results/drl_results.csv` | extract_results | **tab:drl_results**（4 基线 vs 我方） |
| `results/ablation_results.csv` | extract_results | **tab:ablation**（4 变体全 25 算例） |
| `results/decision_time_grid.csv` | extract_results | **fig:decision_time_heatmap** 数据 |
| `results/wilcoxon_results.csv` | extract_results | **tab:wilcoxon**（8 组对比） |
| `results/scalability_summary.csv` | scalability_experiment | **tab:scalability / tab:scalability_time** |
| `results/scalability_wilcoxon.txt` | scalability_experiment | 5.7 行文内检验值 |
| `results/figures/*.pdf`（11 张，含 gantt\_comparison/improvement\_dumbbell/gap\_ecdf） | analyze_training_logs | 论文全部实验配图 |

训练日志 CSV 关键列：`instance_id`（当期算例）、`visit_index`（第几次访问）、`epoch_time_sec`（周期耗时）、`steps_collected`（rollout 决策数）、`eval_steps / eval_solve_time_sec / eval_time_per_decision_ms`（评估决策数/求解耗时/逐决策延迟）、`gap_to_best_rule_percent`、`exploration_epsilon`（DQN/DDQN）。

## 冒烟验证与测试

```bash
# 单元/冒烟测试（含轮询训练器与 DRL 轮询的进程内测试）
python -m pytest tests -q

# 3 算例 × 9 周期端到端冒烟（约 1 分钟）
python multi_train.py --instances u2_t20,u2_t40,u2_t60 --max-epochs 9 \
    --hidden-dim 32 --hgnn-layers 1 --n-trajectories 2 --no-visdom

# 基线冒烟
python -m drl_baselines.multi_run --algorithm A2C \
    --instances u2_t20,u2_t40 --max-epochs 4 --hidden-dim 16 --no-visdom

# 泛化实验冒烟（随机初始化网络走通全流程，无需训练好的 checkpoint）
python scalability_experiment.py --smoke
```

## 文件索引

| 文件 | 职责 |
|---|---|
| `multi_train.py` | **轮询多算例训练器**（我方 + 消融变体）与规则基线缓存 |
| `ablation_experiment.py` | 三个消融变体的轮询训练驱动 |
| `drl_baselines/multi_run.py` | DRL 基线轮询训练驱动 |
| `drl_baselines/base.py` | 基线共享轮询训练循环 `train_round_robin` |
| `drl_baselines/{a2c,dqn,ddqn,reinforce}.py` | 各基线的 `train_epoch` 与单例训练 |
| `drl_baselines/common.py` | 网络构建、计时评估、零样本评测辅助 |
| `scalability_experiment.py` | 大算例生成 + 全方法零样本计时评测 |
| `extract_results.py` | 训练日志 → 论文表格 CSV（last-10 统计 + Wilcoxon） |
| `analyze_training_logs.py` | 训练日志/结果 CSV → 论文全部配图（PDF） |
| `main.py` | 单例训练参考实现、确定性评估、计时协议、Gantt |
| `ppo.py / hgnn.py / mlp.py` | PPO 智能体、HGNN 编码器、Pairwise 策略头 |
| `env.py` | 事件驱动调度环境（自动换电、归一化奖励） |
| `parallel_rollout.py` | 多进程并行轨迹采集（实例逐调用传入） |
| `scheduling_rules.py` | 5 条调度规则（含求解计时） |
| `instance_generator.py` | 算例生成/读写与电池定容 |
| `training_logger.py` | 统一训练日志 schema（67 列） |
| `stats_utils.py` | Wilcoxon 符号秩检验 |
| `config.py` | 分层配置 |
| `baseline_protocol.py` | 统一结果记录（训练耗时/求解耗时语义见 docstring） |
| `metaheuristic_baselines/` | 元启发式占位（未实现，不进入论文结果） |

> 说明：`main.py` 的单例训练（`protocol='single'`）保留为参考与调试入口；论文结果一律来自轮询协议（`protocol='round_robin'`）。
