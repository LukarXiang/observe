# 2026-10-28 Git 发布与续接说明（第108批检查点）

本文件不被元数据引用，修改它不影响 `export_research_metadata.py verify`。

## 本次上传了什么

- 代码、测试、脚本、文档：第65至107批的 `scripts/`、`tests/`、`src/observe/`（`execution.py`、`ledger/book.py`、`strategy_catalog.py`）、`docs/tasks/`、`docs/handoff/` 的验收小文件。
- 元数据只上传当前检查点引用的最小集合：
  - `data/metadata/index.json`
  - `states/4183819e1d15d24907c34add`（当前状态，571文件）
  - `exports/6e3babb7a4d799a74687724a`（清单）
  - `catalogs/0dd173dfd03cb7c148a3`（策略目录）
  - `progress/20261011-080516-0903`、`progress/20261011-081112-2e6e`
- 状态检查点 `20261011-081112-2e6e`，正式验收检查点 `20261011-080516-0903`。

## 刻意没有上传的内容

- 其余93个旧 `states`、93个旧 `exports`、43个旧 `catalogs`、92个旧 `progress`：约 820MB，只读保留在原机 `data/metadata/` 下；Git 不追踪。续接不需要它们，当前索引不引用。
- 新增的 `docs/handoff/` 文件中超过500KB的原始验证日志（本次无；`2026-10-08-batch71-verification.json` 约1MB但被元数据引用，故已上传）。
- `repo/` 与 `data/` 其余载荷：按 AGENTS.md 另包迁移。
- 与 `T-metadata-growth` 一致：不通过删档或重写Git历史解决增长问题，新导出稳定对象共享去重仍待设计。

## 本次验证

- `ruff check src tests scripts`：通过。
- 以暂存区导出的干净副本运行 `scripts/export_research_metadata.py verify --root .`：`ok`，1171个文件，无问题（`payload_verified=false`，仅元数据范围）。
- 全量 pytest（Windows，`D:\envs\quant\.venv`）：开启 `PYTHONUTF8=1` 时 2931 通过、50 失败、187 错误、3 跳过。
  - 未开启 UTF-8 时多出112个失败，原因是测试读取 UTF-8 文件使用 Windows 默认 GBK；Windows 上运行请设置 `PYTHONUTF8=1`。
  - 剩余失败与错误集中在 `test_batch75`–`83_research`、`test_batch91/94/95/96` 等，错误为 `Bound bytes changed: data/catalog/strategies/implementation-evidence.json`、`Source/AST changed`、`Reviewed source set differs`、`Reference changed`、`Frozen91 source changed` 等冻结绑定检查，以及 2 个 Windows 创建符号链接权限错误（WinError 1314）。这些检查要求 `repo/` 原始策略和 `data/` 冻结载荷存在且字节一致；本机载荷与冻结路径（WSL `/mnt/d/...`）不一致，**未逐个确认**，在迁移载荷后的环境（WSL 或恢复载荷后）应重新运行。不要因此修改冻结脚本或正式凭据。
- 为运行测试向既有环境安装 `ta-lib==0.8.1`（`pyproject.toml` 的 `indicators` 可选依赖，未改锁文件）。

## 数据限制与后续起点

- 第108批（两个股票候选，232/340行初读）未正式验收；当前快照缺 `index_constituents`，先查原交接与成员SHA。
- 正式来源研究 300/695，剩 395；交易回测来源12、策略父运行87；严格历史财务可用0。
- 阻断项和待回复决策见 `docs/10-项目状态与续接.md` 的“下一步与阻断”以及 `data/metadata/index.json` 的 decisions/blockers 索引（`D-lowpb81-proxy`、`D-allweather-fill` 等保持 pending）。
- 重构分析与后续策略工作按 AGENTS.md 开头的阅读顺序进行；有 `repo/` 与数据载荷包的机器才能做完整离线复现。
