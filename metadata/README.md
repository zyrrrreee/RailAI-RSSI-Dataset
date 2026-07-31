# Metadata

正式数据版本应提供以下全局目录：

- `scenarios.parquet`：所有场景及稳定基础设施参数；
- `runs.parquet`：所有运行及速度、故障、随机状态、版本信息；
- `samples.parquet`：所有样本窗口、标签和数据位置；
- `field_dictionary.csv`：字段名、类型、单位和公开含义；
- `label_dictionary.csv`：稳定英文标签、中文显示名和故障作用范围；
- `schema/`：机器可读的数据模式。

小型 CSV 示例位于 `data/sample/metadata/`。

当前 `metadata/field_dictionary.csv` 与 `data/sample/` 共同描述仓库可执行的
`v0.2-dev` 示例契约。下一版完整字段契约草案位于
[`docs/schema/`](../docs/schema/README.md)，其中新增了 `world_id`、字段变化维度、
第一版实现阶段与 CSV/Parquet 序列化规则。草案确认前不要单独替换本目录文件，
否则会造成字段字典、生成器、验证器和示例数据互相不一致。
