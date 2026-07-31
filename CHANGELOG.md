# Changelog

## 0.3.0-dev — 2026-07-31

- 增加下一版字段契约草案 Excel 和四份机器可读 CSV；
- 固定 `world_id` 属于 Run、`faulty_ap_id` 属于 Run、`target_ap_id` 属于 Sample、`serving_ap_id` 属于 Observation；
- 将模糊的“随时间变化”改为“记录变化维度”，明确时刻、AP、OBM 三个变化轴；
- 为字段增加第一版核心、条件、可选、后续扩展、派生输出和发布审计标记；
- 增加 CSV/Parquet 序列化与空值规则；
- 当前生成器与示例数据继续使用 `v0.2-dev` 契约，待草案确认后统一迁移。

## 0.2.0-dev — 2026-07-28

- 固定第一版“一条 Run、一个目标 AP、一种状态、一个主要 Sample”契约；
- 明确 `target_ap_id`、候选 `ap_id`、`serving_ap_id` 和 `is_serving` 的区别；
- 将仓库示例升级为 3 AP × 2 OBM 的多链路长表；
- 将同分布测试集明确命名为 `test_id`，与按 Scenario 隔离的 `test_ood` 区分；
- 校验器改为按 AP—OBM 链路检查时间，并增加层级、标签、服务 AP 和划分泄漏检查；
- 定长转换器要求显式选择单条 AP—OBM 链路，避免无提示混合不同物理链路；
- 将无单位 `severity` 拆为故障参数名、数值和物理单位；
- 统一默认生成器、示例数据中的 AP/OBM 编号和标签字典，并补充 Python 版本与完整 CI 测试入口。

## 0.1.0-dev — 2026-07-27

- 建立公开数据集仓库草稿；
- 固定 `Scenario → Run → Sample → Observations` 追溯结构；
- 增加示例长表、元数据目录和固定数据划分文件；
- 接入现阶段 RSSI 生成器基线与仓库验证入口；
- 明确完整数据通过 Release 资产发布，不把大型数据直接提交到 Git 历史；
- 保留仿真证据边界和许可证待确认事项。
