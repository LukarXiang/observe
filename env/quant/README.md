# quant 环境清单

> 以下为早期 Windows / WSL 环境归档。2026-10-04 起跨机续接使用仓库根目录 `pyproject.toml` / `uv.lock`，具体步骤见 [工作总结与跨机续接](../../docs/tasks/2026-10-04-工作总结与跨机续接.md)。不要用本目录旧锁文件覆盖当前根锁文件。Mac 使用根项目 `.venv`；Windows / WSL 可由 uv 在各自目录重建独立项目环境。

Windows 与 WSL 使用同一份 `pyproject.toml` / `uv.lock` 约束，但各自维护独立环境：

| 环境 | 解释器 | 用途 |
| --- | --- | --- |
| Windows | `D:\envs\quant\.venv\Scripts\python.exe` | Windows 数据探测与生产验证 |
| WSL | `~/envs/quant/.venv/bin/python` | WSL 开发、测试、静态检查 |

WSL 环境已于 2026-09-28 用 uv 创建，实际解释器为 Python 3.12.12。仓库通过 `~/projects/observe` 访问（该路径指向 Windows 项目目录），`observe` 以 editable 方式安装。

2026-10-05 已核实 Windows Python 3.12.11、WSL Python 3.12.12，已有环境核心依赖与根锁文件一致，直接使用即可。需要重建 WSL 环境时，在仓库根目录显式指定外部环境，以根锁文件同步：

```bash
UV_PROJECT_ENVIRONMENT="$HOME/envs/quant/.venv" uv sync --frozen --python 3.12.12 --extra ml --extra data --dev
```

在仓库根目录运行 WSL 验证：

```bash
~/envs/quant/.venv/bin/python -m pytest -q
~/envs/quant/.venv/bin/ruff check src tests
```

Windows/WSL 不在仓库目录创建 `.venv`，各平台不共享虚拟环境目录。依赖有变化时，更新仓库根 `pyproject.toml` / `uv.lock`，再显式指定外部 `UV_PROJECT_ENVIRONMENT` 同步；本目录旧锁文件仅保留为历史归档。
