# GitHub 仓库结构

| 路径 | 作用 |
|---|---|
| `configs/` | 可复现的场景和批量生成配置 |
| `data/sample/` | 可直接查看和测试的小样例 |
| `metadata/` | 字段字典、模式和正式元数据说明 |
| `splits/` | 固定训练、验证、测试和 OOD 清单 |
| `generator/` | 仿真生成器基线代码 |
| `scripts/` | 生成、验证、转换入口 |
| `baselines/` | 分类基线及评价协议 |
| `tests/` | 数据契约和生成器测试 |
| `docs/` | 方法、参数、故障定义和引用依据 |
| `checksums/` | Release 文件清单和哈希 |

大型完整数据采用 GitHub Releases 或 Zenodo，不直接塞进 Git 历史。这样可以让代码版本清楚、克隆仓库轻量，并为论文提供稳定 DOI。

