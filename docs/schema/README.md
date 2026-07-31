# RailAI RSSI 字段契约草案 v0.1

本目录保存下一版公开数据契约的审阅材料。当前状态为 **design draft**：它用于统一概念、字段归属、单位、约束和序列化方式，还没有替换仓库中 `data/sample/` 与 `metadata/field_dictionary.csv` 所使用的 `v0.2-dev` 可执行示例契约。

第一版主任务定义为：已知待诊断 `target_ap_id`，输入目标 AP、相邻候选 AP 与车头/车尾两个 OBM 的 RSSI 时序上下文，输出目标 AP 的状态（健康或五类单故障之一）。目标 AP 自动定位属于后续扩展。

## 文件

- `RailAI_RSSI_Field_Dictionary_v0.1-draft.xlsx`：适合团队审阅的完整字段字典；
- `field_dictionary_v0.1-draft.csv`：字段总表的机器可读导出；
- `table_structure_v0.1-draft.csv`：各数据表的粒度、主键和父级关系；
- `enums_and_constraints_v0.1-draft.csv`：枚举值与关键完整性约束；
- `serialization_rules_v0.1-draft.csv`：CSV 与 Parquet 的统一表示规则。

## 关键字段归属

| 字段 | 所属层/表 | 变化维度 | 含义 |
| --- | --- | --- | --- |
| `world_id` | Run / `runs` | 按 Run 固定 | 健康与故障配对的统一随机世界；同一 world 必须整体进入同一数据划分 |
| `faulty_ap_id` | Run / `runs` | 按 Run 固定 | 仿真真值中实际发生故障的 AP；健康 Run 为空 |
| `target_ap_id` | Sample / `samples` | 按 Sample 固定 | 当前要求模型诊断的 AP |
| `ap_id` | `observations` | 按候选 AP | 当前观测行对应的候选 AP |
| `obm_id` | `observations` | 按 OBM | 当前观测行对应的车载接收端 |
| `serving_ap_id` | `observations` | 随时刻 × OBM | 该时刻该 OBM 实际接入的 AP |

`Observations` 是 Sample 内部的逐时刻长表，不是第四个管理层级。原始长表是权威数据；定长矩阵、掩码和统计特征均为可重复生成的派生输出。

## 第一版边界

- 一条故障 Run 只允许一个 `faulty_ap_id`，一条 Run 只生成一个主要 Sample；
- 第一版故障 Run 中主要 Sample 的 `target_ap_id == faulty_ap_id`；
- 健康与五类故障在同一 `world_id` 下共享线路、速度轨迹和配对规则指定的随机传播条件；
- 所有划分以 `world_id` 为最小完整分组，禁止健康/故障配对样本跨集合；
- 故障真值、随机种子和 world 标识不得作为分类模型特征；
- 先完成小规模配对验证和可辨识性检查，再生成大规模数据。

## 迁移顺序

1. 团队和指导老师确认本草案；
2. 将草案冻结为正式 schema 版本；
3. 同步修改生成器、验证器、示例数据和数据卡；
4. 运行契约测试与小规模配对实验；
5. 验证通过后再批量生成并发布数据集。

在第 3 步完成前，请不要用本草案的 139 个字段去解释当前 `data/sample/` 中的少量示例 CSV。
