# quant 环境清单（副本）

实际环境在 `D:\envs\quant`，由 uv 项目方式管理；这里是它的 `pyproject.toml` 与 `uv.lock` 副本，用于重建与追溯版本。

重建步骤：

1. 在 `D:\envs\quant` 放入这两个文件（`pyproject.toml` 里 observe 以相对路径 `../../projects/observe` 可编辑安装，仓库位置不同需改路径）；
2. 在该目录执行 `uv sync`（Python 3.12，会自动安装本仓库为可编辑包）；
3. 验证：`D:\envs\quant\.venv\Scripts\python.exe -m pytest -q`（在仓库根目录执行）。

依赖有变化时，在 `D:\envs\quant` 用 `uv add` / `uv remove`，然后把这两个文件重新复制到这里并提交。
