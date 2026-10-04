# observe

个人 A 股量化研究项目：冻结数据快照、因子与模型研究、规则策略、统一交易账本及离线复现。

当前工作交接见 [2026-10-04 工作总结与跨机续接](docs/tasks/2026-10-04-工作总结与跨机续接.md)，完整执行计划见 [财务数据标准化与策略逐批复现](docs/tasks/2026-10-03-财务数据标准化与策略逐批复现-执行计划.md)。其他设计文档见 [文档索引](docs/README.md)。

根目录 `pyproject.toml` / `uv.lock` 是当前开发环境依据；使用 uv 管理 Python 3.12。先恢复交接文档要求的 `data/` 和策略来源，再运行：

```bash
uv sync --frozen --python 3.12.14 --extra ml --extra data --dev
uv run --frozen --extra ml --extra data pytest -q
uv run --frozen --extra ml --extra data observe --root data data status
```

`data/`、`repo/` 由 `.gitignore` 排除，Git 克隆不会带来行情、财务原始文件、快照或历史实验。跨机应额外复制交接数据包；无需复制 `.venv`、`node_modules` 或任务数据库。

已实现的 5 个规则策略属于固定区间的受限工程复现。财报历史修订版本、供应商周频成分日期精度等限制仍存在，不能据此认定原策略已完整复现或有效。
