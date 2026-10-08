# observe

> **LLM / 新读者入口**：先读 [`AGENTS.md`](AGENTS.md)，再读 [`docs/10-项目状态与续接.md`](docs/10-项目状态与续接.md)，然后沿 [`data/metadata/index.json`](data/metadata/index.json) 读取任务、决策、实验和阻断。Git 只含代码、文档和元数据；行情、完整财务数据与实验载荷在独立数据包中，仅凭 Git 无法重算回测。

个人 A 股量化研究项目：冻结数据快照、因子与模型研究、规则策略、统一交易账本及离线复现。

最新入口：[项目状态与续接](docs/10-项目状态与续接.md)。机器入口为 [研究元数据索引](data/metadata/index.json)，包含完整策略目录、实验摘要、任务、批准、阻断、财务字典和证据指纹。仅克隆 Git 即可阅读这些内容。财务路径、导入和查询见 [财务数据导入与使用](docs/08-财务数据导入与使用.md)。

当前工作交接见 [2026-10-04 工作总结与跨机续接](docs/tasks/2026-10-04-工作总结与跨机续接.md)，完整执行计划见 [财务数据标准化与策略逐批复现](docs/tasks/2026-10-03-财务数据标准化与策略逐批复现-执行计划.md)。其他设计文档见 [文档索引](docs/README.md)。

根目录 `pyproject.toml` / `uv.lock` 是当前开发环境依据；各平台路径见 [技术选型与版本](docs/03-技术选型与版本.md)。本机 Windows 使用 `D:\envs\quant\.venv\Scripts\python.exe`，WSL 使用 `~/envs/quant/.venv/bin/python`，直接调用既有环境；uv 同步时显式指定外部 `UV_PROJECT_ENVIRONMENT`。

以下重建命令适用于 Mac 项目 `.venv`。先恢复交接文档要求的 `data/` 和策略来源，再运行：

```bash
uv sync --frozen --python 3.12.14 --extra ml --extra data --extra indicators --dev
.venv/bin/python scripts/export_research_metadata.py verify
.venv/bin/python -m pytest -q
.venv/bin/observe --root data data status
```

Git 追踪 `data/metadata/` 中的白名单导出，`repo/` 和其他 `data/` 载荷仍排除。元数据是阅读及核验凭据，不作为 Store 的活跃发布状态或回测输入。完整离线复现还需复制对应数据包，再执行 `observe --root data runs index`。管理范围见 [Git追踪计划](docs/tasks/2026-10-08-Git追踪与跨机数据管理计划.md)，实际清单、两包恢复及验证见 [本次实施与交付](docs/tasks/2026-10-08-Git元数据管理实施与交付.md)；无需迁移环境或任务数据库。

当前登记695份来源，人工审查213份，其中211份正式验收；12份来源有本机真实回测及匹配复现。14个代表配置和67个策略父运行不是已完成策略数。原策略完整等价复现为0，财报历史版本、平台语义和固定池偏差仍限制结论；详见状态页。
