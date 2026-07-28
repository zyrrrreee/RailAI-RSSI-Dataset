# Official splits

正式版本提供四个固定清单：

- `train.csv`
- `validation.csv`
- `test_id.csv`
- `test_ood.csv`

每个文件列出 `scenario_id,run_id,sample_id`。

- `test_id`：同一 Scenario 可以出现在训练集，但测试 Run 必须是新的，用于评价已知线路环境下的新运行；
- `test_ood`：整个 Scenario 只能位于域外测试集，用于评价未见线路、AP 布局和传播环境；
- 同一 Run 及其全部 Observation 不能跨集合；
- 使用同一冻结随机世界的健康/故障配对 Run 必须位于同一集合；
- 不能逐 Observation 随机切分。
