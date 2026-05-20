# 考虑能耗约束的多无人船协同任务调度

本项目研究多无人船（USV）协同任务调度问题：多艘无人船从原点充电站出发，在航行能耗、作业能耗、电池容量和自动返航充电约束下完成一组任务，优化目标为最小化所有无人船完成任务并返回后的最大完工时间（makespan）。

当前代码采用“公开固定算例 + 启发式规则基线 + Pairwise Actor + PPO + deterministic evaluation”的训练框架。根据近期训练结果，项目已移除行为克隆预训练逻辑，默认从随机初始化策略直接进行 PPO 训练。

## 当前实验目标

投稿实验的核心目标为：在 25 个公开 CSV 算例上，PPO 收敛后的 deterministic 最优 checkpoint 的多随机种子均值低于该实例的最优调度规则，并在 25 个实例上通过 Wilcoxon signed-rank test（单侧，`PPO < Best Rule`，`p < 0.05`）。

训练前会自动运行以下调度规则作为基线：

- `MinBattery_NearestTask`
- `MaxBattery_NearestTask`
- `NearestOrigin_NearestTask`
- `FarthestOrigin_NearestTask`
- `Random`

所有规则均只运行 1 次；`best_rule_makespan` 取这 5 个单次运行结果中的最优值。`Random` 不再重复运行，也不再取多次随机结果中的最小值。

## 算法框架

```text
Public CSV instance
        |
        v
USV scheduling environment
        |
        v
HGNN actor encoder  -> Pairwise Actor -> legal (task, usv) distribution
HGNN critic encoder -> Critic         -> V(s)
        |
        v
PPO update + deterministic evaluation checkpoint
```

新版 Actor 直接对每个合法 `(task, usv)` 动作对打分：

```text
score(t, u) = MLP([task_embed[t], usv_embed[u], edge_feat[u,t], graph_embed])
```

然后对所有合法 pair 做联合 softmax，PPO 的 `action`、`log_prob` 和 `entropy` 均基于该合法 pair 分布计算。这样可以避免“先选任务、再选无人船”造成的局部匹配信息丢失。

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

训练需要 PyTorch；若只运行启发式规则，只需基础科学计算依赖。

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

Visdom 环境会按实例自动命名，例如：

```text
usv_training_u4_t60
```

主要曲线包括：

- `Train Makespan`
- `Eval Makespan`
- `Best Rule Makespan`
- `Random Makespan`
- `Gap vs Best Rule (%)`
- `Actor Loss`
- `Critic Loss`
- `Entropy`

## 启发式规则基线

运行：

```bash
python scheduling_rules.py
```

脚本会读取当前配置对应的 public 算例，评估全部调度规则，并在 `results/` 保存甘特图。

## 25 个公开算例批量实验

运行完整 public25 实验：

```bash
python public25_experiment.py --max-epochs 500 --seeds 0,1,2,3,4 --visdom
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

## 主要文件

```text
config.py              配置定义
env.py                 调度环境、能耗约束、自动充电、状态与奖励归一化
hgnn.py                异构图神经网络编码器
mlp.py                 Pairwise Actor 与 Critic
ppo.py                 PPO 智能体
main.py                单实例训练、评估、Visdom 和基线集成
public25_experiment.py 25 个公开算例批量训练与统计检验
scheduling_rules.py    启发式调度规则
instance_generator.py  public CSV 算例生成与电池容量验证
utils.py               CSV 算例加载与甘特图绘制
data/public/           25 个公开 CSV 算例
```

## 后续论文实验建议

建议主表报告：

- `Best Rule`
- `Random`
- `PPO`
- `Gap (%)`
- 多种子均值与标准差
- Wilcoxon signed-rank test 的 `p_value`

建议消融实验包括：

- Pairwise Actor vs 旧层级 Actor
- 有/无状态与奖励归一化
- 不同 `n_trajectories`
- 不同 `entropy_coef`
- 不同电池容量 safety factor

