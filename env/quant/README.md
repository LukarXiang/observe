# quant 环境清单

Windows 与 WSL 使用同一份 `pyproject.toml` / `uv.lock` 约束，但各自维护独立环境：

| 环境 | 解释器 | 用途 |
| --- | --- | --- |
| Windows | `D:\envs\quant\.venv\Scripts\python.exe` | Windows 数据探测与生产验证 |
| WSL | `~/envs/quant/.venv/bin/python` | WSL 开发、测试、静态检查 |

WSL 环境已于 2026-09-28 用 uv 创建，实际解释器为 Python 3.12.12。仓库通过 `~/projects/observe` 访问（该路径指向 Windows 项目目录），`observe` 以 editable 方式安装。

重建 WSL 环境：

```bash
mkdir -p ~/envs/quant
cp env/quant/pyproject.toml env/quant/uv.lock ~/envs/quant/
uv sync --project ~/envs/quant --python 3.12
```

在仓库根目录运行 WSL 验证：

```bash
~/envs/quant/.venv/bin/python -m pytest -q
~/envs/quant/.venv/bin/ruff check src tests
```

不要在仓库目录创建或使用 `.venv`。依赖有变化时，在对应环境项目目录用 `uv add` / `uv remove`，再同步提交本目录的 `pyproject.toml` 与 `uv.lock`。
