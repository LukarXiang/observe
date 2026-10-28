# observe

最新正式验收为[第 107 批](docs/tasks/2026-10-11-第一百零七批ETF动量与均线-正式研究验收.md)：300/695、剩 395，12 回测及匹配来源、87 父运行。两 ETF 来源完成全文审查，27 号的 11,663 个原生回归窗口、3,170 个共同排名完成独立核算及禁网匹配；32 本机、32 复制树和 39 正式测试通过。六个缺日、三处价格尺度跳变及 61 号原池缺失保留，新增交易 0。继续剩余来源和基金事件、历史股本、PCF、名称依赖队列，历史目录只读。

进行中：[第 108 批股票候选初读](docs/tasks/2026-10-11-第一百零八批价值低波与月三阳-初读与库存.md)。两份 232/340 行来源已保存，尚未正式验收。当前快照缺成员表，需要先核原交接和历史成员 SHA；国债指数载体、月线与开盘调度仍待核实。

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

当前登记 695 份来源，截至[第 107 批正式研究验收](docs/tasks/2026-10-11-第一百零七批ETF动量与均线-正式研究验收.md)已审查 300 份，剩 395 份；12 份来源有真实回测及匹配复现，策略父运行 87 个。原策略完整等价复现仍为 0。

历史组件及验收记录从[任务索引](docs/tasks/README.md)按批次阅读。当前结果和下一步见[第 107 批报告](docs/tasks/2026-10-11-第一百零七批ETF动量与均线-正式研究验收.md)。最近交易补充见[配对 66 五短岛与 SVM 预检](docs/tasks/2026-10-11-配对66短区间补充回测与SVM零买入预检.md)。

历史版本、平台语义、固定池和交易输入限制保留；未回复经济变体保持pending。全库目标继续，Git由用户提交；[元数据增长审查](docs/tasks/2026-10-09-元数据增长审查.md)要求保留历史并控制新增重复。
