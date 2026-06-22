# vSFC Lab Experiment Scaffold

面向边缘计算中虚拟服务功能链（vSFC）部署/映射实验的可复现代码骨架。

## 特性
- 配置驱动（YAML）+ CLI 参数覆盖
- 多随机种子重复实验
- 结果自动落盘到 `results/<exp_id>/`
- 每次实验保存配置快照、日志、每次运行 JSONL、汇总 CSV
- 绘图脚本读取汇总 CSV（图形配置可扩展）
- 保留 MILP 基线，并通过配置选择算法和指标实现
- 核心科研逻辑提供可扩展接口（`Solver` / `Metrics` / `Topology`）

## 快速开始

```bash
cd vsfc-lab
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m vsfc_lab.cli --config configs/base.yaml --seeds 0 1 2
python scripts/plot.py --summary-csv results/<exp_id>/summary.csv --output results/<exp_id>/fig_overview.png
python scripts/make_scenario.py --output-dir results/scenarios --seeds 0 1 2 --max-nodes 10
```

## 目录结构

```text
vsfc-lab/
  configs/
    base.yaml
  scripts/
    plot.py
    make_scenario.py
  src/vsfc_lab/
    cli.py
    config.py
    registry.py
    runner.py
    io.py
    interfaces.py
    models.py
    mock_components.py
  results/                # 运行后自动生成
```

## 留白位置（你后续填写）
- `src/vsfc_lab/interfaces.py`
  - `Topology` 的具体读取/生成与查询逻辑
  - `Solver.solve(...)` 算法实现
  - `Metrics.evaluate(...)` 指标实现
- `src/vsfc_lab/mock_components.py`
  - 当前包含可运行 MILP 基线、基础网络指标、示例拓扑/请求构造
- `src/vsfc_lab/registry.py`
  - 将新增算法类注册为 `solver_name`
  - 将新增指标类注册为 `metrics_name`

## 当前代码结构分析

- 实验入口：`src/vsfc_lab/cli.py`（也可使用安装后的 `vsfc-run` 命令）。
- 核心算法模块：`src/vsfc_lab/mock_components.py` 中的 `PlaceholderSolver` 是现有 MILP 基线；后续新增算法应实现 `src/vsfc_lab/interfaces.py` 中的 `Solver` 接口。
- 参数配置：`configs/base.yaml` 定义 `solver_name`、`metrics_name`、`topology_name`、`run_kwargs`、输出目录和绘图字段；CLI 的 `--set key=value` 可覆盖配置。
- 结果输出：`src/vsfc_lab/runner.py` 为每次实验保存 `config.snapshot.json`、`runs.jsonl`、`summary.csv` 和 `run.log`。
- 画图逻辑：`scripts/plot.py` 从 `summary.csv` 读取指标并输出图片。
- 场景复现：`scripts/make_scenario.py` 可生成固定随机种子的 topology/request JSON，CLI 可用 `--scenario-config` 复用场景。

## 算法过程提供后再实现

在收到具体算法过程后，修改代码前应先复述：

1. 算法目标：优化吞吐量、时延、成本、资源利用率、公平性或其他目标。
2. 输入：拓扑、链路容量/时延、节点资源、SFC 请求、VNF 链、随机种子和算法超参数。
3. 输出：每个请求的 VNF 放置、路径映射、卸载/未服务流量，以及算法元数据。
4. 核心步骤：初始化、决策变量/状态、迭代或求解过程、约束处理、停止条件。
5. 可能歧义：目标函数权重、不可行请求处理方式、流量缩放、链路方向、随机过程和对比基线。

未收到具体算法过程时，不应伪造新算法或实验结果。

## 算法与指标选择

`configs/base.yaml` 默认使用：

```yaml
solver_name: "milp_baseline"
metrics_name: "network_metrics"
```

当前可选名称：

- `solver_name`: `milp_baseline`（兼容别名：`placeholder_solver`）
- `metrics_name`: `network_metrics`（兼容别名：`placeholder_metrics`）

新增算法时，建议新增一个实现 `Solver.solve(...)` 的类，然后在 `src/vsfc_lab/registry.py` 的 `SOLVER_FACTORIES` 中注册名称。这样可以通过配置或命令行选择运行：

```bash
python -m vsfc_lab.cli --config configs/base.yaml --seeds 0 1 2 --set solver_name=my_algorithm
```

## 已有实验指标

`network_metrics` 会输出可用于网络实验对比的基础指标：

- `objective_total`：MILP 基线目标函数重构值
- `throughput`：成功处理的总流量
- `avg_end_to_end_delay_ms`：平均端到端路径时延
- `offloading_ratio`：卸载/未本地处理流量占比
- `avg_link_utilization`、`max_link_utilization`：链路利用率
- `jain_fairness`：基于各请求成功处理流量的 Jain 公平性
- `algorithm_runtime_sec`：算法运行时间
- `avg_offloading_flow`、`n_solutions`：辅助统计

## 复现实验建议
- 固定 `--seeds` 序列并记录配置快照
- 每次实验使用唯一 `exp_id`
- 使用 `summary.csv` 作为论文表格和绘图数据源

## 输出结果路径

每次运行默认输出到 `results/<exp_id>/`：

- `config.snapshot.json`：运行配置、随机种子和 CLI 参数快照
- `runs.jsonl`：每个 seed 的状态、指标和解
- `summary.csv`：跨 seed 的均值、标准差和样本数
- `run.log`：运行日志
- `fig_overview.png`：可由 `scripts/plot.py` 生成的汇总图

## 结果分析模板（论文撰写）

可在真实实验运行后按以下结构撰写：

1. 实验设置：说明拓扑来源、请求规模、VNF 链、随机种子、算法参数和硬件/软件环境。
2. 对比方法：说明新增算法与 `milp_baseline` 或其他基线的区别。
3. 主要结果：对比吞吐量、平均时延、链路利用率、公平性和算法运行时间。
4. 现象解释：分析资源约束、链路瓶颈、流量卸载比例和路径选择对结果的影响。
5. 稳定性：报告多随机种子的均值与标准差，说明算法对场景扰动的敏感性。
6. 局限性：说明当前实验规模、拓扑抽象、流量模型或求解时间限制。
7. 后续扩展：增加真实流量、更多拓扑、动态请求、收敛曲线和参数敏感性实验。
