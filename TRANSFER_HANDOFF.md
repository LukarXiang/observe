# Observe 项目迁移交接

这份说明对应当前迁移包：

- 代码提交：`c765f2cbb35f8279a80527b590ea61271d17eef0`
- 代码版本：`c765f2c add the normalized Ridge penalty mode and run the frozen controlled comparison`
- 已发布数据批次：`20260930-121727-4c66`
- 数据归档：`observe-data-full-20261001.tar`

## 归档内容

归档包含整个 `data/` 目录，覆盖：

- `data/PUBLISHED.json`：当前发布状态
- `data/std/`：标准化 Parquet 分区，包括日线、复权、公司行动、复权覆盖、5 分钟线和分钟股票池
- `data/snapshots/`：已有数据快照
- `data/runs/`：已有研究、回测、成对实验、复现、重新评价和诊断产物
- `data/batches/`、`data/audits/`、`data/minute_audits/`、`data/coverage/`：发布与审计记录
- `data/A股分钟线/`、`data/raw/`、`data/staging/`、`data/probe/`：原始输入、导入中间文件和探测记录

`data/locks/` 是运行时锁目录，归档时排除了其中的锁文件。新电脑启动任务时会自动创建所需锁文件。

## 新电脑安装与解压

在新电脑上执行：

```bash
git clone https://github.com/LukarXiang/observe.git
cd observe
git checkout c765f2cbb35f8279a80527b590ea61271d17eef0

# 将归档文件放在当前目录后解压；归档内包含 data/ 前缀
tar -xf observe-data-full-20261001.tar

# Python 3.12+ 与 uv
uv sync --dev
```

如果归档不在项目目录，可指定绝对路径：

```bash
tar -xf /path/to/observe-data-full-20261001.tar -C /path/to/observe
```

Windows 上可使用 7-Zip 解压 `.tar` 文件，目标目录必须是仓库根目录，使解压后存在 `observe/data/PUBLISHED.json`。

## 解压后的校验

```bash
test -f data/PUBLISHED.json
test -f data/std/calendar/all__d6b85c54.parquet
test -f data/std/instruments/all__6dbb75b5.parquet
test -f data/std/adj_factors/all__5517c69d.parquet
test -f data/std/adj_coverage/all__de373d9c.parquet
test -f data/std/bars_5m/202609__3a34204d.parquet
test -f data/std/minute_universe/all__02ecf18c.parquet
git rev-parse HEAD
uv run pytest
```

`git rev-parse HEAD` 应输出上述代码提交。`data/` 被 `.gitignore` 排除，不要把大数据目录提交到 Git。

## 使用方式

运行现有数据中心、回测或研究时始终使用同一个数据根目录：

```bash
uv run observe --root data <subcommand>
```

使用已发布状态时读取 `data/PUBLISHED.json`。使用旧实验或复现时，必须保留对应的 `data/runs/<run_id>/` 和 `data/snapshots/`，并使用实验目录中记录的快照与配置。复现命令禁止联网，缺少快照或标准分区时应先报告缺失，不要自动下载替代数据。

分钟特征成对实验依赖已发布的 `bars_5m` 和 `minute_universe`。只有重新导入或更新分钟原始数据时，才需要 `data/A股分钟线/` 原始压缩包。

## 给新电脑 Agent 的接手提示词

```text
你正在接手 observe 项目。工作目录是仓库根目录，数据已从 observe-data-full-20261001.tar 解压到 data/。

先阅读 TRANSFER_HANDOFF.md、CONTEXT.md 和相关模块文档。先执行以下检查，不要删除、清理或重新下载 data/：

1. 确认 git commit 是 c765f2cbb35f8279a80527b590ea61271d17eef0。
2. 确认 data/PUBLISHED.json 存在，发布批次是 20260930-121727-4c66。
3. 确认 PUBLISHED.json 引用的 data/std 分区都存在。
4. 执行 uv sync --dev，然后执行 uv run pytest。
5. 任何研究、回测或复现都使用 --root data，并优先读取已发布状态或实验记录的 snapshot_id。

数据目录是本地实验输入，不要提交到 Git，也不要运行 observe gc --apply、删除 snapshots/runs、覆盖 PUBLISHED.json 或联网补数，除非用户明确要求。

如果任务是复现实验：先读取 data/runs/<run_id>/manifest.json、config.json 或 config.yaml、data_manifest.json，按记录的 snapshot_id 和配置执行 reproduce；如果哈希、快照或分区缺失，停止并报告具体路径。

如果任务是分钟成对实验：确认已发布状态包含 bars_5m 和 minute_universe；不要把 data/A股分钟线 原始压缩包当作运行时必需输入。只有用户要求重新导入分钟源数据时才使用它。

报告问题时给出命令、具体文件路径、发布批次和实验 run_id，不要用新的联网数据静默替换现有数据。
```
