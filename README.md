# vSFC Lab Experiment Scaffold

面向边缘计算中虚拟服务功能链（vSFC）部署/映射实验的可复现代码骨架。

## 特性
- 配置驱动（YAML）+ CLI 参数覆盖
- 多随机种子重复实验
- 结果自动落盘到 `results/<exp_id>/`
- 每次实验保存配置快照、日志、每次运行 JSONL、汇总 CSV
- 绘图脚本读取汇总 CSV（图形配置可扩展）
- 核心科研逻辑全部留白接口（`Solver` / `Metrics` / `Topology` 细节）

## 快速开始

```bash
cd /path/to/vsfc-lab
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
  - 示例拓扑/请求构造仅为占位，可替换为你的数据输入

## 复现实验建议
- 固定 `--seeds` 序列并记录配置快照
- 每次实验使用唯一 `exp_id`
- 使用 `summary.csv` 作为论文表格和绘图数据源

## 当前仓库结构分析
- 实验入口：`src/vsfc_lab/cli.py`，读取 YAML 配置和 CLI 覆盖项，按 `solver_name` / `metrics_name` 选择实验组件。
- 核心算法模块：`src/vsfc_lab/mock_components.py`，当前保留 `milp_baseline`（兼容旧名 `placeholder_solver`）作为 MILP 基线。
- 参数配置：`configs/base.yaml`，可用 `--set key=value` 或 `--set run_kwargs.time_limit=30` 覆盖。
- 结果输出：`src/vsfc_lab/runner.py` 将配置快照、逐次运行 JSONL、汇总 CSV 和日志保存到 `results/<exp_id>/`。
- 画图逻辑：`scripts/plot.py` 从 `summary.csv` 读取指标并输出图像。

## 新算法接入约定
请先提供具体算法过程。实现前应明确复述：
1. 算法目标：例如优化 vSFC 映射成本、吞吐量、时延或公平性。
2. 输入：拓扑、链路容量/时延、SFC 请求、VNF 资源需求、随机种子和算法参数。
3. 输出：`PlacementSolution` 列表，包括 VNF 到节点的映射、路径、卸载/未服务流量和元数据。
4. 核心步骤：初始化、候选选择、约束检查、迭代/收敛条件、结果生成。
5. 可能歧义：不可行请求如何处理、卸载是否等价丢包、时延模型（当前仅统计链路传播时延，不含 VNF 处理时延）、链路方向、是否允许共享节点等。

接入时在 `src/vsfc_lab/mock_components.py` 或新模块中实现 `Solver.solve(...)`，再在 `src/vsfc_lab/cli.py` 的 `SOLVER_REGISTRY` 注册名称，即可通过配置选择：

```bash
python -m vsfc_lab.cli --config configs/base.yaml --seeds 0 1 2 --set solver_name=milp_baseline
```

## 已支持的实验指标
`network_metrics`（兼容旧名 `placeholder_metrics`）输出：
- `throughput`：成功处理的总流量。
- `avg_end_to_end_delay_ms`：按路径累计的平均端到端时延。
- `packet_loss_rate`：未处理/卸载流量占输入流量比例。
- `avg_link_utilization` / `max_link_utilization`：链路利用率均值和最大值。
- `fairness_index`：按请求处理流量计算的 Jain 公平性指数。
- `algorithm_runtime_seconds`：单次 seed 的算法运行时间。
- `objective_total`、`avg_offloading_flow`、`n_solutions`：保留的基线指标。

## 输出与可复现性
固定 `--seeds` 后，实验会保存：
- `results/<exp_id>/config.snapshot.json`：合并后的配置、随机种子、可选组件列表。
- `results/<exp_id>/runs.jsonl`：每个 seed 的指标和解。
- `results/<exp_id>/summary.csv`：跨 seed 的均值、标准差和样本数。
- `results/<exp_id>/run.log`：运行日志。

运行器会在每次调用算法前执行 `random.seed(seed)`，并把同一 seed 通过 `Solver.solve(..., seed=seed)` 传入；如新增算法需要隔离随机性，建议在算法内部基于该 seed 创建局部随机数生成器。
当前时延指标与代码实现保持一致：只累计链路传播时延，VNF 处理时延暂按 0 处理，后续可在指标实现中扩展。

## 论文结果分析模板
可按以下结构撰写实验分析：
1. 实验设置：说明拓扑来源、请求规模、VNF 链、随机种子、硬件环境和关键参数。
2. 对比方法：列出 `milp_baseline` 与新增算法，说明二者输入一致。
3. 性能指标：报告吞吐量、时延、丢包率、链路利用率、公平性和运行时间的均值/标准差。
4. 结果讨论：分析新增算法相对基线的提升、代价和适用场景。
5. 威胁与局限：说明随机拓扑、数据规模、不可行实例处理等限制。
6. 可扩展方向：增加真实流量、更多拓扑规模、动态请求到达、不同 VNF 链长度和参数敏感性实验。
