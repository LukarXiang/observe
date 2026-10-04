# observe 开发与续接

开始前阅读 `docs/tasks/2026-10-04-工作总结与跨机续接.md`，再按 `docs/tasks/2026-10-03-财务数据标准化与策略逐批复现-执行计划.md` 推进。计划前半部是启动时盘点，当前完成情况以第 11、12 节和交接记录为准。

本机环境规范是 `/Users/zuozhe/Setups/tool/Mac开发环境与Coding-Agent工具链建设规范_v1.0.md`；该文件可访问时，编码前读取并遵循。其他电脑同样使用 uv 管理 Python，以根目录 `pyproject.toml` / `uv.lock` 重建环境；归档 `env/quant` 的旧锁文件不能替代当前根锁文件。前端使用 Node.js 24 LTS，Mac npm prefix 为 `~/.local`。安装前核实工具存在与任务需求；不要使用系统 pip、Conda、sudo、清理或重置命令。修改全局配置必须先读取和备份。

- `repo/`、`data/` 不在 Git 中。开始研究前检查交接清单、快照分区和原始策略来源是否齐全；禁止把缺失输入悄悄换成最新下载结果。
- 旧快照、不可变分区、冻结参数、历史实验和 manifest 保持只读。新增证据通过新批次、新快照和新实验登记；实验目录不能覆盖。
- 复现已有实验使用 `observe reproduce <run_id>`；迁移后先 `observe --root data runs index` 重建本机路径登记库，冻结 JSON 中的旧路径不改写。
- 财报期末不是可用日期；最终整理值不证明历史版本；实收资本金额不是股数；年度控制变量市值不是每日总市值。严格历史财务模式仍排除未证明版本的记录。
- 中证800名单为供应商周频归档，`strict_usable=false`。月间保留的目标可能已经不属于当日指数；检查选股日，不能用月间名单变化否定原冻结目标。
- 复用唯一账本，不另建费用/净值实现。所有新策略保留来源 SHA256、规则审查、明确忠实程度、成本情景和离线复现证据。
- 下一批先核实历史股本事件、PCF 和历史证券名称/股票池；未补齐依赖前不宣称微盘400或小盘价值100已经完成。

验证命令：

```bash
uv run --frozen --extra ml --extra data ruff check src tests scripts/prepare_handoff.py
uv run --frozen --extra ml --extra data pytest -q
git diff --check
```

按变化运行必要验证，无新问题时避免重复全量检查。代码及文档更新须写明本次验证、数据限制和后续起点。
