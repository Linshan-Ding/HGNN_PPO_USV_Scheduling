# 考虑能耗约束的多无人船协同任务调度

本项目研究多无人船（USV）协同任务调度问题：多艘无人船从原点充电站出发，在航行能耗、作业能耗、电池容量和自动返航充电约束下完成一组任务，优化目标为最小化所有无人船完成任务并返回后的最大完工时间（makespan）。

当前主算法为 `Pairwise Actor + PPO`。项目保留简单调度规则基线，并新增两个对比算法集目录：`drl_baselines/` 用于组织深度强化学习对比算法，`metaheuristic_baselines/` 用于组织元启发式对比算法。DRL 对比算法已有单实例训练入口；未完整实现的算法默认不会进入正式 public25 实验结果。

## 当前实验目标

投稿实验的核心目标为：在 25 个公开 CSV 算例上，PPO 收敛后的 deterministic 最优 checkpoint 的多随机种子均值低于该实例的最优调度规则，并在 25 个实例上通过 Wilcoxon signed-rank test（单侧，`PPO < Best Rule`，`p < 0.05`）。

训练前会自动运行以下简单调度规则作为基线：

- `MinBattery_NearestTask`
- `MaxBattery_NearestTask`
- `NearestOrigin_NearestTask`
- `FarthestOrigin_NearestTask`
- `Random`

所有规则均只运行 1 次；`best_rule_makespan` 取这 5 个单次运行结果中的最优值。`Random` 不再重复运行，也不再取多次随机结果中的最小值。

## 算法体系

```text
Public CSV instance
        |
        v
USV scheduling environment
        |
        +-- Main method: Pairwise Actor + PPO
        +-- Simple rules: scheduling_rules.py
        +-- DRL baselines: drl_baselines/
        +-- Metaheuristic baselines: metaheuristic_baselines/
```

主算法中的 Actor 直接对每个合法 `(task, usv)` 动作对打分：

```text
score(t, u) = MLP([task_embed[t], usv_embed[u], edge_feat[u,t], graph_embed])
```

然后对所有合法 pair 做联合 softmax，PPO 的 `action`、`log_prob` 和 `entropy` 均基于该合法 pair 分布计算。

## 对比算法集目录

`drl_baselines/` 用于深度强化学习对比算法，当前包含可运行实现：

- `A2C`
- `DQN`
- `DDQN`
- `REINFORCE`

DRL 统一接口：

```python
algorithm.train(instance, cfg)
algorithm.evaluate(instance, cfg)
algorithm.save(path)
algorithm.load(path)
```

`metaheuristic_baselines/` 用于元启发式对比算法，当前包含模板：

- `GA`
- `PSO`
- `ACO`
- `SA`

元启发式模板统一接口：

```python
algorithm.solve(instance, cfg)
```

两类算法都使用 `baseline_protocol.py` 中的 `AlgorithmResult` 作为统一结果格式：

```text
algorithm_name, category, instance_id, n_usvs, n_tasks,
makespan, success, runtime_sec, seed
```

每个算法集目录都有 `registry.py`。默认 `list_algorithms()` 只返回已经实现并验证的算法；DRL 目录当前会返回 `A2C`、`DDQN`、`DQN`、`REINFORCE`，元启发式目录当前仍是占位模板，默认不会进入正式实验。若需要查看全部占位算法，可使用：

```python
from drl_baselines import registry as drl_registry
from metaheuristic_baselines import registry as meta_registry

print(drl_registry.list_algorithms(include_unimplemented=True))
print(meta_registry.list_algorithms(include_unimplemented=True))
```

本轮明确不加入 A3C 作为对比算法，避免引入异步多进程训练带来的工程复杂度和论文口径争议。

单独运行一个 DRL 对比算法：

```bash
python -m drl_baselines.run --algorithm DDQN --n-usvs 2 --n-tasks 20 --max-epochs 100 --seed 0
```

该命令会读取对应 public 算例，训练单个 DRL baseline，并在 `results/` 中保存单实例结果 CSV。
DRL baseline 默认启用 Visdom 实时曲线，env 会自动命名为：

```text
drl_baselines_{algorithm}_{instance_id}
```

例如 `drl_baselines_DDQN_u2_t20`。主要曲线包括 `Train Makespan`、`Eval Makespan`、`Best Eval Makespan`、`Success Rate`，以及算法对应的 `Actor Loss`、`Critic Loss`、`Entropy` 或 `Q Loss`、`Epsilon`。如需关闭 Visdom：

```bash
python -m drl_baselines.run --algorithm DDQN --no-visdom
```

## 公开算例

公开算例位于 `data/public/`，共 25 个：

- 无人船数量：`2, 4, 6, 8, 10`
- 任务数量：`20, 40, 60, 80, 100`
- 文件命名：`u{n_usvs}_t{n_tasks}.csv`

例如：

```text
data/public/u4_t60.csv
data/public/u10_t100.csv
```

训练和规则评估默认只需修改配置中的 `n_usvs` 与 `n_tasks`，即可读取对应 public 算例：

```python
cfg = get_config(
    n_usvs=4,
    n_tasks=60,
    data_dir='data/public',
)
```

CSV 元信息中包含 `battery_capacity`。环境加载实例时使用文件中的电池容量，不会重新随机生成或覆盖。

电池容量按如下规则生成：

```text
duration_j = (t1_j + 2 * t2_j + t3_j) / 4
single_trip_energy_j =
    2 * distance(origin, task_j) * energy_cost_per_distance
    + duration_j * energy_cost_per_task_time
battery_capacity = 1.20 * max(single_trip_energy_j)
```

该规则保证任意单个任务都可以从原点出发、执行并安全返回，其中也包括最远任务。

## 安装依赖

```bash
pip install -r requirements.txt
```

训练主 PPO 需要 PyTorch；若只运行启发式规则或对比算法接口 smoke test，只需基础科学计算依赖。

若需要实时训练曲线，先启动 Visdom：

```bash
python -m visdom.server -port 8097
```

浏览器打开：

```text
http://localhost:8097
```

## 单实例训练

直接运行：

```bash
python main.py
```

默认训练参数建议：

- `lr_encoder=1e-4`
- `lr_actor=3e-4`
- `lr_critic=3e-4`
- `n_trajectories=8`
- `ppo_epochs=4`
- `entropy_coef=0.01`
- `eval_interval=10`

代码调用示例：

```python
from config import get_config
from main import train, evaluate

cfg = get_config(
    n_usvs=4,
    n_tasks=60,
    data_dir='data/public',
    max_epochs=500,
    seed=0,
    hidden_dim=256,
    hgnn_layers=3,
    n_heads=4,
    n_trajectories=8,
    rollout_num_workers=4,
    vectorized_update=True,
    update_batch_size=128,
    ppo_epochs=4,
    lr_encoder=1e-4,
    lr_actor=3e-4,
    lr_critic=3e-4,
    entropy_coef=0.01,
    use_visdom=True,
)

agent, instance, train_info = train(cfg)
evaluate(cfg, agent, instance)
```

训练会在开始前计算启发式基线，并每 `eval_interval` 个 epoch 做一次 deterministic evaluation。checkpoint 只根据 deterministic `Eval Makespan` 保存，不根据训练采样 makespan 保存。

`n_trajectories` 条训练轨迹只使用多进程并行采集。主进程会在每个 epoch 开始时冻结当前策略快照，多个 worker 在 CPU 上并行 rollout，随后主进程合并 transitions 并执行一次 PPO update。`n_trajectories` 必须不小于 2。相关配置：

```python
cfg = get_config(
    rollout_num_workers=0,      # 0 表示 min(n_trajectories, os.cpu_count())
    rollout_device='cpu',
    rollout_torch_threads=1,
)
```

PPO update 默认使用 mini-batch 向量化实现，避免逐 transition 重复执行 HGNN 前向。相关配置：

```python
cfg = get_config(
    vectorized_update=True,
    update_batch_size=128,
    update_shuffle=True,
)
```

如需调试旧版逐样本 update，可在实验入口添加 `--legacy-update`。

训练过程默认会实时追加保存 CSV 日志，每个 run 一个文件：

```text
results/training_logs/{run_id}.csv
```

`run_id` 格式为：

```text
PPO_{variant}_{instance_id}_seed{seed}_{timestamp}
```

CSV 每行对应一个 epoch，核心字段包括：

```text
run_id,algorithm,variant,instance_id,n_usvs,n_tasks,seed,epoch,
timestamp,elapsed_sec,train_reward_avg,train_reward_std,
train_makespan_avg,train_makespan_min,train_makespan_std,
success_rate,n_trajectories,n_success,n_failed,
eval_makespan,eval_success,best_eval_makespan,best_eval_epoch,
gap_to_best_rule_percent,best_rule_name,best_rule_makespan,
random_makespan,actor_loss,critic_loss,entropy,
lr_actor_encoder,lr_actor,lr_critic_encoder,lr_critic,
lr_shared_encoder,hidden_dim,hgnn_layers,n_heads,ppo_epochs,
vectorized_update,update_batch_size,update_shuffle,
gamma,gae_lambda,clip_epsilon,entropy_coef,reward_normalization,
best_model_path,rollout_time_sec,update_time_sec,epoch_time_sec,
batch_prepare_time_sec,actor_update_time_sec,critic_update_time_sec
```

如需降低写入频率，可在配置中设置：

```python
cfg = get_config(
    save_training_csv=True,
    training_log_dir='results/training_logs',
    training_log_interval=5,
)
```

## 25 个公开算例批量实验

运行完整 public25 实验：

```bash
python public25_experiment.py --max-epochs 500 --seeds 0,1,2,3,4 --visdom
```

如需手动控制并行采集：

```bash
python public25_experiment.py --max-epochs 500 --rollout-num-workers 4
python public25_experiment.py --max-epochs 500 --update-batch-size 64
```

输出文件：

- `results/public25_summary.csv`
- `results/public25_wilcoxon.txt`

`public25_summary.csv` 字段：

```text
instance_id,n_usvs,n_tasks,best_rule_name,best_rule_makespan,
ppo_mean,ppo_std,gap_percent,pass_instance
```

其中：

```text
gap_percent = (ppo_mean - best_rule_makespan) / best_rule_makespan * 100
```

投稿目标要求 25 个实例的 `gap_percent` 均为负数，并且 Wilcoxon 单侧检验 `p < 0.05`。

## 消融实验

当前消融实验保留三种核心变体：

- `no_hgnn`：用轻量 MLP/Linear 节点编码器替代 `HGNNEncoder`，保留 Pairwise Actor。
- `shared_encoder`：Actor 与 Critic 共用同一个 `HGNNEncoder`，共享 encoder 只进入一个优化器。
- `no_reward_norm`：保留完整网络结构，但关闭奖励尺度归一化，使用原始 makespan 奖励。

运行单实例消融实验：

```bash
python ablation_experiment.py --variants full,no_hgnn,shared_encoder,no_reward_norm --n-usvs 4 --n-tasks 60 --max-epochs 500 --seeds 0,1,2,3,4
```

输出文件：

```text
results/ablation_summary.csv
```

字段包括：

```text
variant,instance_id,n_usvs,n_tasks,seed,best_eval_makespan,
best_rule_name,best_rule_makespan,gap_to_rule_percent,
gap_to_full_percent,success
```

若需要消融实验的 Visdom 实时曲线，添加 `--visdom`。env 会按变体和算例命名，例如：

```text
usv_ablation_no_hgnn_u4_t60
```

消融实验同样支持并行 rollout：

```bash
python ablation_experiment.py --variants full,no_hgnn,shared_encoder,no_reward_norm --n-usvs 2 --n-tasks 20 --rollout-num-workers 4
```

## 训练日志分析与作图

训练结束后，可基于实时保存的 CSV 做实验分析和候选训练曲线图：

```bash
python analyze_training_logs.py --log-dir results/training_logs --output-dir results/figures
```

该脚本会生成：

- `results/training_logs/summary.csv`：每个 run 的最终最优 evaluation、最优 epoch 和相对规则 gap。
- `results/figures/run_*_curves.png`：单 run 的训练 makespan、eval makespan 和历史最优曲线。
- `results/figures/mean_curve_*.png`：同一实例、同一变体的多随机种子均值曲线。
- `results/figures/ablation_*.png`：同一实例下不同消融变体对比曲线。
- `results/figures/gap_by_tasks.png`：按任务规模汇总的相对最优规则 gap 分布。

常用筛选参数：

```bash
python analyze_training_logs.py --instance-id u4_t60 --variant full
```

## 测试与检查

检查新增对比算法接口：

```bash
python -m unittest tests.test_baseline_interfaces tests.test_ablation_variants tests.test_training_logger tests.test_parallel_rollout tests.test_vectorized_update
```

运行启发式规则基线：

```bash
python scheduling_rules.py
```

检查 Python 文件语法：

```bash
python -m py_compile config.py hgnn.py mlp.py ppo.py main.py public25_experiment.py ablation_experiment.py analyze_training_logs.py training_logger.py parallel_rollout.py
```

## 主要文件

```text
config.py                    配置定义
env.py                       调度环境、能耗约束、自动充电、状态与奖励归一化
hgnn.py                      异构图神经网络编码器
mlp.py                       Pairwise Actor 与 Critic
ppo.py                       PPO 智能体
main.py                      单实例训练、评估、Visdom 和基线集成
public25_experiment.py       25 个公开算例批量训练与统计检验
ablation_experiment.py       三种 PPO 消融变体训练与结果汇总
training_logger.py           训练过程实时 CSV 日志
analyze_training_logs.py     训练日志汇总、筛选与候选曲线图生成
parallel_rollout.py          n_trajectories 多进程并行轨迹采集
scheduling_rules.py          简单启发式调度规则
baseline_protocol.py         对比算法统一结果协议
drl_baselines/               深度强化学习对比算法集
metaheuristic_baselines/     元启发式对比算法集
instance_generator.py        public CSV 算例生成与电池容量验证
utils.py                     CSV 算例加载与甘特图绘制
data/public/                 25 个公开 CSV 算例
```

## 后续论文实验建议

建议主表报告：

- `Best Rule`
- `Random`
- `PPO`
- `Gap (%)`
- 多种子均值与标准差
- Wilcoxon signed-rank test 的 `p_value`

建议后续扩展对比算法时遵循两个原则：

- 只有 `implemented=True` 且通过 smoke test、可复现验证的算法才进入正式论文表。
- 深度强化学习与元启发式算法均输出统一 `AlgorithmResult`，便于汇总到同一 public25 结果表。
