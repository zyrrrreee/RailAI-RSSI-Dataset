# RailAI-RSSI-Dataset

面向车地通信故障诊断的、可复现的铁路 RSSI 仿真数据集与基准代码。

> 当前仓库草稿版本：`v0.2-dev`。现阶段公开的是数据结构、生成器基线、验证方法和少量示例数据，不把仿真数据声称为某条真实线路的实测数据。

## 项目目标

本项目在缺少真实故障 RSSI 数据的条件下，建立一套可解释、可追溯、可重复生成的数据集，支持“输入 RSSI 状态，输出故障类别”的分类研究。当前标签包括：

1. 健康；
2. 全链路功率衰减；
3. 天线 1 功率下降；
4. 天线 2 功率下降；
5. 天线 1 方向偏移；
6. 天线 2 方向偏移。

复合故障属于扩展任务，不与五类单故障强行混为一个单标签类别。

## 数据逻辑

数据采用三级追溯结构：

```text
Scenario（线路与通信环境）
└── Run（一次完整列车运行）
    └── Sample（围绕目标 AP 截取的诊断样本）
        └── Observations（样本内逐时刻/逐位置观测行）
```

- `scenario_id`：描述线路长度、AP 布局、设备与传播环境等相对稳定条件；
- `run_id`：描述一次完整运行中的速度轨迹、`world_id`、随机种子以及健康/故障真值；
- `sample_id`：描述围绕 `target_ap_id` 的一个主要诊断窗口；
- `observations`：保存时间、位置、速度、AP、OBM、RSSI、服务 AP 状态等逐点数据。

第一版公开契约采用“一条 Run 对应一个目标 AP、一种状态和一个主要
Sample”。Sample 仍保留目标 AP、相邻候选 AP、车头/车尾两个 OBM
的观测，不等于只保留一条 RSSI。多 Sample Run 和多目标复合故障属于后续扩展。

详细说明见 [docs/DATASET_STRUCTURE.md](docs/DATASET_STRUCTURE.md)。

## 字段契约草案

下一版字段契约已整理为 Excel 与机器可读 CSV，见
[docs/schema/README.md](docs/schema/README.md)。该目录目前是供团队和老师审阅的
`v0.1-draft`，尚未替换 `data/sample/` 正在使用的 `v0.2-dev` 示例格式。
确认草案后，生成器、验证器、示例数据和数据卡必须在同一次迁移中更新，不能只换字段表。

## 仓库结构

```text
RailAI-RSSI-Dataset/
├── README.md
├── DATASET_CARD.md
├── CITATION.cff
├── LICENSE
├── CHANGELOG.md
├── configs/
├── data/sample/
├── metadata/
├── splits/
├── generator/
├── scripts/
├── baselines/
├── tests/
├── docs/
└── checksums/
```

完整数据不会直接作为大型 Git 文件反复提交。正式版本计划通过 GitHub Releases（后续可同步 Zenodo）发布，并提供 SHA-256 校验文件。

## 快速开始

### 1. 创建环境

要求 Python 3.10 或更高版本，推荐使用 Python 3.11。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. 检查仓库内的示例数据

```powershell
python scripts\validate_dataset.py --root data\sample
```

### 3. 运行当前生成器基线

```powershell
python scripts\generate_dataset.py --output artifacts\rssi_research_v1
```

该入口复用现阶段经过测试的 RSSI 生成器，输出当前研究格式。正式 `Scenario → Run → Sample` 批量导出器将作为后续版本完善，不能把当前兼容输出误称为最终公开数据集。

### 4. 转换为定长模型输入

```powershell
python scripts\convert_to_model_input.py `
  --observations data\sample\observations\sample_demo.csv `
  --ap-id AP-002 `
  --obm-id OBM-front `
  --output artifacts\sample_model_ready.npz `
  --length 256
```

当前转换器一次只转换一条明确的 AP—OBM 链路。若输入包含多条链路但
没有指定 `--ap-id` 和 `--obm-id`，程序会拒绝转换，避免把不同物理链路
悄悄混成一条曲线。输出包含 `X`、`mask`、样本追溯字段和标签。原始长表
始终是权威数据，定长矩阵只是派生表示。

## 数据划分原则

- 官方提供固定的 `train / validation / test_id / test_ood` 清单；
- 同一 `run_id` 不能跨集合；
- 强相关或配对的健康/故障运行不能被拆到不同集合；
- `test_id` 按 Run 隔离，评价已知 Scenario 下的新运行；
- `test_ood` 按 Scenario 隔离，评价未见线路、AP 布局与传播环境；
- 不使用逐行随机划分，否则相邻采样点会造成严重数据泄漏。

## 可复现与可信度边界

当前生成器包含时间域运动轨迹、对数距离路径损耗、双定向天线、空间相关阴影、Rician 小尺度衰落、接收机积分、噪声、量化、灵敏度/饱和处理、双 OBM、多 AP 候选及有状态切换研究代码。

这些机制可以证明代码内部的物理方向、统计规律和数据契约相互一致，但不能证明仿真已经等同于某条目标线路。所有默认参数必须标记为“文献先验、设备先验或工程假设”，并在取得设备资料或实测数据后重新校准。

## 许可证与引用

当前 `v0.2-dev` 尚未完成团队作者信息和双许可证确认，因此 `LICENSE` 暂时保留“发布前待确认”状态。建议正式公开时：

- 代码：MIT License；
- 数据：Creative Commons Attribution 4.0 International（CC BY 4.0）。

正式发布前还需补全 `CITATION.cff` 中的团队成员、指导教师、学校和仓库 URL。
