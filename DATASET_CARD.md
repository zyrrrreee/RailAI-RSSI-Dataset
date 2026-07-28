# Dataset Card

## Dataset summary

RailAI-RSSI-Dataset 是面向铁路车地无线通信故障诊断的仿真 RSSI 数据集。其主要价值不是替代实测数据，而是在真实故障样本难以获取时，提供机制清楚、参数可追溯、能够重复生成的研究基准。

## Intended tasks

- 健康/异常检测；
- 五类单故障分类；
- 故障严重度分析；
- 异常区间定位；
- 多 AP、多 OBM 时序建模；
- 域外场景泛化评估。

## Labels

| label | 含义 | 当前机理 |
|---|---|---|
| healthy | 健康 | 仅含传播与接收机随机性 |
| global_power_attenuation | 全链路功率衰减 | 目标 AP 两副天线共享的公共射频链路预算下降 |
| antenna_1_power_loss | 天线 1 功率下降 | 天线 1 支路功率/增益下降 |
| antenna_2_power_loss | 天线 2 功率下降 | 天线 2 支路功率/增益下降 |
| antenna_1_direction_offset | 天线 1 方向偏移 | 天线 1 方向图旋转 |
| antenna_2_direction_offset | 天线 2 方向偏移 | 天线 2 方向图旋转 |

## Unit of observation

一个 `Sample` 是目标 AP 附近的一段诊断窗口。样本内每行 `Observation` 对应一个报告时刻，并至少包含：

- 时间 `time_s`；
- 线路位置 `position_m`；
- 速度 `speed_mps`；
- `target_ap_id`、候选链路 `ap_id` 与 `obm_id`；
- 接收机上报 `rssi_dbm`；
- 当前 `serving_ap_id` 以及该候选链路是否服务 `is_serving`；
- RSSI 是否有效及接收机状态。

第一版每个 Run 只设置一个目标 AP、一种健康/故障状态，并提取一个主要
Sample。一个 Sample 内保留多个候选 AP 与车头、车尾两个 OBM，所以同一
时间可以合法出现多行 Observation。

## Hierarchy and traceability

每个样本必须可以追溯到：

- `scenario_id`；
- `run_id`；
- `sample_id`；
- 故障标签、目标设备、故障参数名、数值和物理单位；
- 随机种子；
- 生成器版本；
- 配置哈希；
- 配对健康运行（若有）；
- 官方数据划分。

## Generation assumptions

- RSSI 定义为车载 OBM 接收的轨旁 AP 下行信号强度；
- AP 具有两副沿轨道相反方向的定向天线；
- 车头/车尾 OBM 共享空间阴影场，小尺度衰落与接收机噪声可独立；
- 速度通过 `x(t)`、时间采样、接收机时间窗和切换空间距离产生影响，不作为额外 dB 项直接加入路径损耗；
- 当前参数范围来自论文先验、公开资料或工程假设，不代表目标线路实测真值。

## Known limitations

1. 缺少目标线路真实健康与故障 RSSI 进行绝对校准；
2. AP 间距、列车长度、天线安装与接收机参数仍需设备资料确认；
3. 当前五类故障是基础故障集合，不覆盖所有车地通信失效模式；
4. 当前生成器基线仍需完成统一的 `Scenario → Run → Sample` 批量导出；
5. 分类准确率只能评价仿真域内可分性，不能单独证明数据真实。
6. 当前定长转换器只支持显式选择一条 AP—OBM 链路；完整多链路张量布局仍属后续工作。

## Responsible use

本数据集适合科研、教学和算法对比，不应直接用于安全关键铁路系统的上线决策。任何工程部署都必须使用目标线路数据、设备规范和独立安全验证。
