# scripts 说明

这里的脚本分四类。批次脚本被 `data/metadata` 按路径和 SHA 引用，**不要改名、移动或原地修改**；需要新行为时新增脚本。

Windows 用 `D:\envs\quant\.venv\Scripts\python.exe`，WSL 用 `~/envs/quant/.venv/bin/python`，Mac 用项目 `.venv`；不要在仓库内裸跑 `uv run/sync`。

## 日常维护（会被反复使用）

| 脚本 | 作用 |
| --- | --- |
| `export_research_metadata.py` | 导出并校验 `data/metadata/`（`export` / `verify`），状态更新后必跑 |
| `build_strategy_catalog.py` | 生成策略目录与结果表 |
| `prepare_handoff.py` | 按交接计划打包冻结数据并校验搬迁后的文件 |
| `verify_financial_import.py` | 财务导入的离线核对 |
| `verify_offline_tests.py` | 禁网条件下的测试核对 |
| `verify_stage1.py` | 阶段 1 真实数据验收（在市证券数、日线抽样比对、复权连续性） |
| `check_minute_publish.py`、`check_missing_minute_source.py` | 分钟线发布与缺失来源检查 |

## 逐批策略研究（一次性，已冻结）

| 模式 | 作用 |
| --- | --- |
| `review_strategy_batchNN.py`（8–64，部分批次） | 第 NN 批来源的规则审查与证据登记 |
| `run_strategy_batchNN.py`（4–19，部分批次） | 第 NN 批真实回测运行 |
| `verify_strategy_batchNN.py`（3、10、11） | 第 NN 批的独立复核 |
| `probe_strategy_*.py`、`recover_strategy_batch26.py` | 依赖/数据探测与恢复 |
| `archive_strategy_*.py` | 批次产物归档 |

每批的文字记录见 [docs/tasks/README.md](../docs/tasks/README.md)，机器可读凭据在 `docs/handoff/`。

## 原始规则与外部证据核查

`research_*.py`、`extract_*_evidence.py`：ETF510310、聚宽 EMA、沪深 ETF 税费规则、退市与换股结算、披露证据的提取脚本。结论记录在对应的 `docs/tasks/*核查.md`。

## 数据源探测

`probe/`：BaoStock、AKShare、通达信、存储与速度的实测脚本，重跑方法见 `docs/01-数据源实测报告.md` 末尾。
