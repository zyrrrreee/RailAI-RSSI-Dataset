# Sample data

这里仅保存用于解释字段和检查仓库契约的小样例，不代表正式数据集规模或最终参数。
当前示例使用 3 个 AP、车头/车尾 2 个 OBM，并以 `AP-002` 为目标 AP，
用于演示同一时刻多条候选链路、服务 AP 和链路身份的保存方式。

- `metadata/scenarios.csv`：场景；
- `metadata/runs.csv`：完整运行；
- `metadata/samples.csv`：诊断样本；
- `observations/sample_demo.csv`：样本内逐点观测。

正式数据版本将以 Parquet 为主，并通过 GitHub Releases 发布；CSV 只用于小样例和快速查看。
