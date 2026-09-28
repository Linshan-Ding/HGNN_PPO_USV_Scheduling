# 在本机手动运行 GitHub Actions

本工作流专用于 `Linshan-Ding/HGNN_PPO_USV_Scheduling`。代码推送、PR 和结果提交不会自动触发运行。

## 在网页运行

1. 打开仓库 **Actions → Local Windows experiments → Run workflow**。
2. **Use workflow from** 保持 `master`。在 **source_ref** 填要执行的分支（包括 `claude/...`）或完整提交 SHA。
3. 选择任务并点击 **Run workflow**。首次使用选择 `smoke`。

| 参数 | 含义 |
|---|---|
| `task=smoke` | 现有测试、主方法 3 算例 × 9 周期、各消融/基线短训练、随机模型泛化流程检查；这些输出不进入正式批次 |
| `task=train` | 运行 `method` 选择的一种方法；消融也从这里选择 |
| `task=scalability` | 使用训练完成的模型做正式泛化评估，必要 checkpoint 缺失则失败 |
| `task=report` | 从日志提取结果表并出图；缺少可选方法/模型会在摘要标明并跳过相关图 |
| `task=republish` | 将本机已有运行重新上传，不重新计算；填写 `republish_run` |
| `method` | `full`、`no_hgnn`、`shared_encoder`、`no_reward_norm`、`A2C`、`DQN`、`DDQN`、`REINFORCE` |
| `epochs` / `seed` | 正式训练默认 5000 周期 / seed 0；smoke 使用固定短配置 |
| `instances` | 逗号分隔，例如 `u2_t20,u2_t40`；留空为全部 25 算例 |
| `data_source` | 后处理选择 `batch` 本机新实验或 `historical` 所选代码提交中的历史结果 |
| `batch_id` | 训练摘要提供的 24 位批次编号；留空按代码版本与表单参数定位 |
| `republish_run` | 原运行编号和尝试次数，例如 `123456789-1`，见运行摘要 |

默认正式网络参数为 hidden_dim=256、hgnn_layers=3、n_heads=4、n_trajectories=8。训练使用 GPU 更新、CPU rollout，关闭 Visdom。脚本不自动升级依赖。

## 分项实验与汇总

同一代码提交、seed、训练周期和算例子集形成同一个批次；为每种方法分别手动启动一次。可以提前提交多个任务，它们排队串行执行，不取消正在运行的实验。

汇总新实验时填训练摘要中的 `batch_id`，并把 `source_ref` 设为该批次记录的完整代码 SHA。系统只读取该批次每种方法最后一次成功运行，排除失败与冒烟。显式选择批次时使用批次中的参数，不使用表单中无关的默认 seed 等参数。

选择 `historical` 时，只读取所选提交中的 `results` 和 `models`，不合入本机新实验。历史模型缺失时，正式泛化不能运行；出图可跳过缺模型的 Gantt。部分结果报告会注明不完整，不表示已完成全部论文实验。

所有新训练都是从头开始。本配置不实现优化器状态恢复；中断后已有最优模型会保留，但重新启动训练不是断点续训。

## 结果在哪里

- **Actions 运行摘要**：状态、任务、源代码 SHA、批次编号及结果链接。
- **`results` 分支**：`README.md` 是索引，`runs/<运行编号>-<尝试次数>/` 保存报告和单个不超过 10 MiB 的小型结果。
- **Actions Artifacts**：模型、完整训练日志、控制台日志及所有输出，保留 14 天。到期前下载需要长期保存的附件。
- **本机**：`D:\GitHubActions\HGNN_PPO_USV_Scheduling\runs\`。不自动清理；磁盘不足 5 GiB 时拒绝开始计算。

训练日志和模型不会自动加入 Git 历史。仓库当前公开，结果分支和可访问的运行附件也应视为公开科研输出。

计算和发布分为两个 job。计算最长 72 小时，smoke 最长 30 分钟，后处理和发布各最长 2 小时。发布 job 取得新的 GitHub 令牌，因此长训练不依赖开始时的令牌。失败输出也会尝试回传，计算失败不会被发布成功掩盖。

若取消、断网或关机导致附件/提交缺失，待电脑重新上线后选 `republish`，填原运行编号。补传会更新原结果目录和索引的附件链接，不改原计算状态。若从未成功初始化运行，则只有 Actions 日志，没有本机输出可补传。

## 服务与设备

- 运行器安装在 `C:\actions-runner\hgnn-ppo-usv`，标签为 `hgnn-ppo-usv`。
- 服务以 `NT AUTHORITY\NETWORK SERVICE` 运行，自动启动，使用非管理员身份计算。
- Python 固定为 `E:\anaconda3\envs\python3.13\python.exe`。
- 电脑必须开机联网。锁屏不影响后台服务；执行期间抑制系统空闲睡眠，但关机、断电及主动休眠会中断任务。
- 在仓库 **Settings → Actions → Runners** 检查 `Idle` / `Active` / `Offline` 状态。

在管理员 PowerShell 中，可通过名称筛选这个仓库的专用服务：

```powershell
Get-Service -Name 'actions.runner.Linshan-Ding-HGNN_PPO_USV_Scheduling.*'
# 将上一步得到的精确服务名用于下列命令
Stop-Service -Name '<精确服务名>'
Start-Service -Name '<精确服务名>'
```

在 GitHub 运行页面点击 **Cancel workflow** 可停止任务。仅终止该任务的子进程，不结束其他 Python 程序。

## 维护与访问边界

入口固定由仓库所有者从 `master` 手动触发，重新运行也检查操作者身份。请选择自己确认可信的代码提交；自托管运行器不是隔离沙箱。不要给工作流添加执行陌生 fork 或外部 PR 的触发器。

运行器保留自动更新。不要在仓库或日志中保存注册令牌；注册使用 GitHub 提供的短期令牌。运行时采用自动生成的 `GITHUB_TOKEN`，计算 job 只读，发布 job 才有仓库写入权限。

排查顺序：检查 Runner 在线状态 → 打开失败步骤日志 → 查看本机对应运行目录的 `manifest.json` 和 `logs` → 修复代码或环境 → 按需重新运行或补传。不要通过删除整个输出根目录来修复单次失败。
