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
cd /root/exp/vsfc-lab
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
