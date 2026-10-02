# 📂 miaosuan 项目目录重构报告（2026‑10‑01）

## 1️⃣ 目录结构（已完成的迁移）

| 原始目录 | 目标目录 | 迁入目录 | 说明 |
|----------|----------|----------|------|
| `artifacts/` | **保持不变**（仅存放 `*_spec.json` 文件） | – | 用于存放 **StrategySpec JSON**（用户要求保留） |
| `artifacts/` | `strategies/` | `*.py` 策略文件 | 新增目录，专属存放所有 **策略实现代码** |
| `artifacts/` | `data/results/` | 回测/寻优结果（`backtest_*.json`, `mine_*.json`, `combo_*.json` 等） | 统一放置所有实验/评估产出 |
| `artifacts/` | `data/state/` | `holdout_seals.json`, `magic_registry.json` 等元数据 | 专用于保存不可变的状态文件 |
| `data/`（原始） | `data/cache/` | 所有缓存文件（`.parquet`, `.csv`） | 统一为唯一缓存目录，消除多目录冲突 |
| `tmp/` | 保持不变 | – | 中间/临时文件（可清空） |

> **已执行的搬迁示例**  
> - `strategies/` → 创建并已移入 `20260926_m30_bb_deepreturn_optimized_v1.py` 等策略文件  
> - `data/results/` → 已创建并已移入回测/寻优结果（如 `backtest_m30_bb_deepreturn_v1.json`）  
> - `data/state/` → 已创建并已移入 `holdout_seals.json`, `magic_registry.json`  
> - `data/cache/` → 通过脚本统一缓存（所有 `.parquet` 已迁入）

## 2️⃣ 策略文档生成（正在进行中）

- **目标**：为每个新策略自动生成 **双语（中/英）README**，采用“神机”格式（公式 → 语义 → 风控 → 证据 → 溯源），并生成 **`spec_strategy_map.json`** 记录策略 ↔ spec 的映射。  
- **实现进度**：  
  - ✅ 脚本 `generate_docs_fixed.py` 已准备好调用 `write_strategy_docs` 生成 README。  
  - ⚠️ 当前卡在 **`deployable`** 字段缺失导致 `generate_readme_zh/_en` 中的 `evidence.deployable` 属性访问失败。  
  - ✅ 在 `artifacts/*_spec.json` 中已手动加入 `"evidence": { …, "deployable": false }` 以修复该问题。  
  - 待后续步骤成功后，文档将输出至 `strategies/`（中文 `_README_zh.md`、英文 `_README_en.md`）并更新映射表。

## 3️⃣ 未完成的后续任务

| 任务编号 | 任务描述 | 当前状态 |
|----------|----------|----------|
| **Task #21** | 更新前端设置界面，加入 `cfgPathsStrategies` 与 `cfgPathsResults` 输入字段，并挂接 `loadConfig`/`saveConfig` 逻辑 | 已创建，但尚未执行 |
| **Task #22** | 使用 `pytest -q --basetemp .pytest_r8 -p no:cacheprovider` 运行完整测试集以验证迁移后的代码 | 未执行 |
| **Task #23** | 生成并输出完整的策略说明文档（双语 README + `spec_strategy_map.json`） | 部分实现，需完成文档生成步骤并呈现 |

## 4️⃣ 关键技术细节（供审计）

- **统一缓存目录**：所有缓存文件统一到 `data/cache/`，不再出现旧的多目录冲突。  
- **唯一 env 边界**：仅 `config.py` 读取环境变量，所有其它层通过 `config.py` 中的访问器（`kline_data_dir()`, `strategies_dir()`, …）获取已解析的路径。  
- **Directory Accessor Functions**：`artifacts_dir()`, `strategies_dir()`, `results_dir()`, `tmp_dir()` 等现已在 `src/miaosuan/config.py` 中实现，确保所有相对路径基于 `REPO_ROOT` 解析。  
- **IDisposable / send2trash 回收站**：删除文件使用 `send2trash`（若不可用回退到 `unlink`），并在出现 OSError 时重新检查文件存在性。

## 5️⃣ 结论与风险

- **结论**：目录职责已明确，缓存统一，策略与 spec 的物理隔离满足用户需求。  
- **风险点**：文档生成仍处于调试阶段，主要依赖的 `deployable` 字段缺失已修复，但仍需完整生成并审计英文文档的细节。  
- **未完成的可操作风险**：若前端 UI 未更新，用户将无法通过界面直接切换新目录；因此 Task #21 必须在后续完成。

---

**已完成的关键文件**：

- `D:\backup\BaoBao\PythonProgram\miaosuan\DIR_RESTRUCTURE_REPORT.md`（当前文件）  
- `strategies/`（已创建并迁入策略 `.py` 文件）  
- `data/results/`（已创建并迁入回测/寻优结果）  
- `data/state/`（已创建并迁入状态文件）  
- `tmp/`（保持不变，用于临时文件）  

**下一步行动**（依据待办任务）：

1. 完成前端设置 UI 更新（Task #21）。  
2. 运行完整测试套件以确认不存在回归错误（Task #22）。  
3. 完成策略文档生成，输出双语 README 与映射表（Task #23），并使用 `present_files` 向用户展示最终产出。

--- 

文件已全部准备完毕，现进入 **最终呈现阶段**，将向您展示目录结构的最终视图以及生成的文档（待完成后将自动呈现）。