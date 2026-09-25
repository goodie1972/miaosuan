# Spec 管理方案设计

> 解决四个核心问题：导出决策机制、spec↔.py 映射、谱系追踪与版本管理、全生命周期支撑。

---

## 一、现状诊断

| 维度 | 现状 | 缺口 |
|------|------|------|
| spec 存储 | `artifacts/` 下散落 JSON，无索引 | 无 manifest，发现靠 glob 扫描 |
| spec ↔ .py 映射 | .py 注释里写了 `spec_id`（纯文本），ExportResult 不含 spec_id | 无反向索引，无法从 spec 查到 .py |
| 谱系追踪 | Provenance 只有 git_sha/data_fingerprint/seed，无 parent_spec_id | tune 不产出派生 spec，无 lineage 链 |
| 导出决策 | 人工运行 `export` 命令，不看门禁 | 无自动化决策，BLOCKED 也能导出（但标 RESEARCH_ONLY） |
| magic 账本 | 只记 (name, version) → magic | 不记 spec_id、.py 路径、导出时刻 |

---

## 二、总体架构

```
                         ┌──────────────────────────────────┐
                         │         SpecRegistry             │  ← 新增
                         │  (artifacts/spec_registry.json)   │
                         │  spec_id → {路径, 状态, 谱系, …}  │
                         └────────┬───────────┬──────────────┘
                                  │           │
          ┌──────────┐    mine    │           │  export     ┌──────────┐
          │ 行情数据  │──────► Spec ──────► Registry ────► Strategy .py
          └──────────┘           │           │   ↕ 映射      └──────────┘
                                 │           │                  ↕
                          backtest│     tune  │             verify
                                 ▼           ▼
                          ┌──────────────────────────┐
                          │   artifacts/backtest/    │
                          │   artifacts/tune/        │
                          └──────────────────────────┘
```

**核心原则：SpecRegistry 是唯一索引层，不改 spec 本身的四段结构（payload/semantics/evidence/provenance），只在 spec 外部维护元数据。**

---

## 三、SpecRegistry 设计

### 3.1 存储位置

```
artifacts/spec_registry.json
```

与 `magic_registry.json`、`holdout_seals.json` 并列，属于 artifacts 级持久台账。

### 3.2 数据结构

```jsonc
{
  "version": 1,
  "specs": [
    {
      "spec_id": "5d0841b6db9fdca2",         // 内容寻址 ID（已有）
      "name": "h1_xauusd_miaosuan",
      "file_path": "artifacts/spec.json",     // 最新 spec 文件的相对路径
      "gate_verdict": "BLOCKED",              // 门禁快照
      "deployable": false,
      "created_at": "2026-09-16T04:58:27+00:00",
      "market": "FOREX_XAUUSD",
      "budget": "quick",
      "data_fingerprint": "sha256:43c4...",
      "git_sha": "42b89187",
      "seed": 20260910,
      // —— 谱系 ——
      "parent_spec_id": null,                  // mine 产出 → null；tune 派生 → 父 spec_id
      "lineage": [],                           // 完整谱系链 [root_id, …, parent_id]
      "derivation": "mine",                    // "mine" | "tune" | "manual"
      // —— 生命周期 ——
      "status": "ACTIVE",                      // "ACTIVE" | "ARCHIVED" | "DEPRECATED"
      "exports": [                              // 该 spec 的所有导出记录
        {
          "py_path": "shenji-strategies/20260913_h1_testxau_miaosuan_v1.py",
          "magic": 661801,
          "exported_at": "2026-09-13T10:00:00+00:00",
          "lint_ok": true,
          "deployable": false                  // 导出时刻的门禁状态
        }
      ],
      "backtests": [                            // 该 spec 的所有回测记录
        {
          "result_path": "artifacts/backtest/backtest_20260920_101211.json",
          "run_at": "2026-09-20T10:12:11+00:00",
          "sharpe": 4.70,
          "max_drawdown": 0.12
        }
      ],
      "tunes": [                                // 该 spec 的所有寻优记录
        {
          "result_path": "artifacts/tune/tune_abc12345.json",
          "run_at": "2026-09-21T15:00:00+00:00",
          "best_score": 2.1,
          "child_spec_id": "a3f8c2d1e9b70451"  // 派生的新 spec_id（如有）
        }
      ]
    }
  ]
}
```

### 3.3 操作接口

```python
class SpecRegistry:
    """Spec 索引台账（artifacts/spec_registry.json 的 Python 包装）。"""

    def register(self, spec: StrategySpec, *, derivation: str = "mine",
                 parent_spec_id: str | None = None) -> SpecRecord:
        """注册一个 spec（mine 或 tune 产出后调用）。

        - 首次注册：创建记录，计算 lineage = parent.lineage + [parent_id] 或 []
        - 重复注册（spec_id 已存在）：更新 file_path 和 gate_verdict，保留 exports/backtests/tunes
        """

    def record_export(self, spec_id: str, *, py_path: str, magic: int,
                      lint_ok: bool, deployable: bool) -> None:
        """记录一次导出事件。"""

    def record_backtest(self, spec_id: str, *, result_path: str,
                        sharpe: float, max_drawdown: float) -> None:
        """记录一次回测事件。"""

    def record_tune(self, spec_id: str, *, result_path: str,
                    best_score: float, child_spec_id: str | None = None) -> None:
        """记录一次寻优事件（child_spec_id 在 tune 产出派生 spec 时填入）。"""

    def find_by_spec_id(self, spec_id: str) -> SpecRecord | None:
        """O(1) 查找。"""

    def find_by_magic(self, magic: int) -> SpecRecord | None:
        """从 magic 反查到 spec（跨过 .py 中间层）。"""

    def find_by_py_path(self, py_path: str) -> SpecRecord | None:
        """从 .py 路径反查到 spec。"""

    def list_lineage(self, spec_id: str) -> list[SpecRecord]:
        """返回完整谱系链（从根到当前）。"""

    def list_children(self, spec_id: str) -> list[SpecRecord]:
        """返回所有直接派生子 spec。"""

    def archive(self, spec_id: str) -> None:
        """归档（不再出现在默认列表中，但不删除文件）。"""

    def rebuild_from_disk(self) -> int:
        """从 artifacts/ 全量重建索引（灾难恢复）。"""
```

### 3.4 写入安全

- 文件锁复用 `magic_registry.py` 已有的 `_ledger_lock` 机制
- 写盘时 `sort_keys=True` + 紧凑分隔（与 `codec.py` 一致），保证 diff 友好
- 每次 `mine`/`export`/`backtest`/`tune` 命令结束时更新 registry

---

## 四、问题 1：导出决策机制

### 4.1 当前问题

- export 命令不看门禁，BLOCKED 也能导出（只标 RESEARCH_ONLY）
- 没有自动触发机制，全靠人工手动运行

### 4.2 设计：三层决策模型

```
┌───────────────────────────────────────────────────────┐
│                  导出决策模型                          │
│                                                       │
│  Layer 1: 代码质量门（lint）       ← 硬拦，已有        │
│  Layer 2: 策略质量门（gate）       ← 软标，已有        │
│  Layer 3: 导出准入规则（policy）   ← 新增              │
└───────────────────────────────────────────────────────┘
```

**Layer 1（lint）和 Layer 2（gate）保持不变**——lint ERROR 拒绝导出，gate 结论嵌入文件。新增 **Layer 3：导出准入规则**，决定"要不要导出"和"导出成什么状态"。

### 4.3 导出准入规则（ExportPolicy）

```python
@dataclass(frozen=True)
class ExportPolicy:
    """导出准入规则（可配置）。"""

    # BLOCKED 的 spec 是否允许导出为 RESEARCH_ONLY
    allow_blocked_research: bool = True    # 默认允许（研究用）

    # DEPLOYABLE 的 spec 是否自动触发导出
    auto_export_deployable: bool = False   # 默认不自动（需人工确认）

    # RESEARCH_ONLY 的 spec 是否自动触发导出
    auto_export_research: bool = False     # 默认不自动

    # 同一 spec_id 是否允许重复导出（True = 每次都写新 .py；False = 幂等）
    allow_reexport: bool = False           # 默认幂等（同 spec 已导出则跳过）

    # 最大 RESEARCH_ONLY 文件数（防止堆积）
    max_research_files: int = 20
```

### 4.4 导出触发场景

| 场景 | 触发者 | 规则 | 产出 |
|------|--------|------|------|
| **人工导出** | 用户运行 `export --spec X` | lint 通过即导出；gate 结论嵌入文件 | .py + registry 记录 |
| **mine 后自动导出** | mine 命令结束时（可选 `--auto-export` flag） | gate=DEPLOYABLE 且 `auto_export_deployable=True` 时自动导出 | .py + registry 记录 |
| **tune 后自动导出** | tune 命令产出派生 spec 后（可选 `--auto-export` flag） | 同上 | .py + registry 记录 |
| **Web UI 一键导出** | UI 按钮触发 | 同人工导出 | .py + registry 记录 |

### 4.5 导出幂等性

```python
def should_export(registry: SpecRegistry, spec_id: str, policy: ExportPolicy) -> tuple[bool, str]:
    """判断是否应该导出，返回 (是否导出, 原因)。"""
    record = registry.find_by_spec_id(spec_id)
    if record is None:
        return False, "spec 未注册"

    if record.status == "ARCHIVED":
        return False, "spec 已归档"

    if not policy.allow_reexport and record.exports:
        last = record.exports[-1]
        if Path(last.py_path).exists():
            return False, f"已导出至 {last.py_path}（幂等跳过）"

    return True, "允许导出"
```

---

## 五、问题 2：spec ↔ .py 映射

### 5.1 映射的维护时机

在 `cli.py` 的 `export` 命令中，写完 .py 文件后，调用 `registry.record_export()`：

```python
# cli.py export 命令末尾（在 write_text 之后追加）
registry = SpecRegistry()
registry.record_export(
    spec_id=spec.spec_id,
    py_path=str(target),
    magic=result.magic or 0,
    lint_ok=result.ok,
    deployable=spec.evidence.deployable,
)
```

### 5.2 双向查询路径

```
正向：spec_id → registry.specs[id].exports[-1].py_path → .py 文件
反向：.py 路径 → registry.find_by_py_path() → spec_id → artifacts/spec.json
间接：magic → registry.find_by_magic() → spec_id → .py + spec.json
```

### 5.3 .py 文件内嵌的结构化标记

当前 .py 注释里已有 `spec_id`（纯文本）。**新增一个结构化常量**，方便 verify 和自动解析：

```python
# 在模板 factor_kernel_v1.py.j2 的常量区新增
SPEC_ID = "{{ spec_id }}"        # 内容寻址 ID，可反查 registry
SOURCE_SPEC = "{{ spec_file_path | default('') }}"  # 源 spec 文件路径（可选）
```

lint 不需要新增规则（AF 系列只管代码质量），但 verify 命令可以利用 `SPEC_ID` 常量做反向校验：

```python
# verify 命令增强：检查 .py 内嵌的 spec_id 与 registry 记录一致
def verify_with_registry(py_path: str, registry: SpecRegistry) -> None:
    source = Path(py_path).read_text()
    spec_id = _extract_const(source, "SPEC_ID")
    record = registry.find_by_spec_id(spec_id)
    if record is None:
        warn(f"spec_id {spec_id} 未在 registry 中注册")
    elif not any(e.py_path == py_path for e in record.exports):
        warn(f".py 文件不在 spec {spec_id} 的导出记录中")
```

### 5.4 magic 账本增强

在 `MagicLedger.allocate()` 的 record 中追加 `spec_id` 字段：

```jsonc
// magic_registry.json（增强后）
{
  "records": [
    {
      "magic": 661801,
      "name": "h1_xauusd_miaosuan",
      "version": 1,
      "spec_id": "5d0841b6db9fdca2",   // ← 新增
      "allocated_at": "2026-09-13T10:00:00+00:00"
    }
  ]
}
```

这样 magic → spec_id → spec.json 形成完整链路。

---

## 六、问题 3：谱系追踪与版本管理

### 6.1 Provenance 增强

在 `Provenance` dataclass 中新增两个可选字段（向后兼容，旧 spec 反序列化时为 None/空）：

```python
@dataclass(frozen=True)
class Provenance:
    # ... 现有字段不变 ...

    # —— 谱系（新增，可选，缺省为 None/空，向后兼容）——
    parent_spec_id: str | None = None      # 父 spec 的内容寻址 ID
    lineage: tuple[str, ...] = ()           # 完整谱系链 [root_id, ..., parent_id]
    derivation: str = "mine"                # 派生方式："mine" | "tune" | "manual"
```

`to_dict()` 和 `from_dict()` 相应增加这三个字段的序列化/反序列化。旧 spec JSON 没有这些字段时，`from_dict` 用 `.get()` 默认值，**完全向后兼容**。

### 6.2 谱系构建规则

| 派生方式 | parent_spec_id | lineage | 触发时机 |
|----------|---------------|---------|----------|
| **mine**（首次挖掘） | `None` | `[]` | mine 命令产出新 spec |
| **mine**（同因子重跑） | `None` | `[]` | spec_id 相同（内容寻址），registry 更新不新增 |
| **tune**（参数寻优产出派生 spec） | 源 spec_id | `parent.lineage + [parent_id]` | tune 产出新 spec |
| **manual**（手工编辑后另存） | 源 spec_id | `parent.lineage + [parent_id]` | 手动操作 |

### 6.3 tune 产出派生 spec（关键改造）

当前 tune 只产出 `TuneResult` JSON（参数组合），不产出新 spec。改造为：

```python
# tune/engine.py TuneEngine.optimize() 末尾新增

def build_tuned_spec(source_spec: StrategySpec, best_params: dict) -> StrategySpec:
    """用 tune 最优参数构造派生 spec。

    - payload 不变（tokens 相同）
    - semantics 用最优参数覆盖（neutral_band, roll_window, long_short 等）
    - evidence 重新计算（对全样本跑一次回测 → 更新 sharpe/dsr/gate_verdict）
    - provenance 追加 parent_spec_id + lineage + derivation="tune"
    """
    new_semantics = replace(source_spec.semantics,
        neutral_band=float(best_params.get("neutral_band", source_spec.semantics.neutral_band)),
        roll_window=int(best_params.get("roll_window", source_spec.semantics.roll_window)),
        long_short=bool(best_params.get("long_short", source_spec.semantics.long_short)),
    )
    # 重新评估门禁
    new_evidence = re_evaluate_gate(source_spec.tokens, new_semantics, ...)
    new_provenance = replace(source_spec.provenance,
        parent_spec_id=source_spec.spec_id,
        lineage=source_spec.provenance.lineage + (source_spec.spec_id,),
        derivation="tune",
        created_at=now_utc(),
    )
    return StrategySpec(
        spec_version=source_spec.spec_version,
        name=f"{source_spec.name}_tuned",
        payload=source_spec.payload,           # 不变
        semantics=new_semantics,              # 更新
        evidence=new_evidence,                # 重新评估
        provenance=new_provenance,            # 追加谱系
        notes=f"tuned from {source_spec.spec_id}",
    )
```

### 6.4 谱系可视化示例

```
mine → spec A (id=5d0841, BLOCKED, dsr=0.036)
         │
         ├── tune → spec B (id=a3f8c2, RESEARCH_ONLY, dsr=0.72)
         │            │
         │            └── tune → spec C (id=b7e1d9, DEPLOYABLE, dsr=0.96)
         │
         └── mine（更多数据重跑）→ spec D (id=c2f3a1, DEPLOYABLE, dsr=1.12)

lineage(A) = []
lineage(B) = [A]
lineage(C) = [A, B]
lineage(D) = []               # 全新挖掘，无父 spec
```

### 6.5 版本管理

spec 的"版本"有两个维度：

| 维度 | 机制 | 说明 |
|------|------|------|
| **因子版本** | `spec_id`（内容寻址） | tokens 或 semantics 变了 → spec_id 变 → 新版本 |
| **导出版本** | `STRATEGY_VERSION` + magic 尾两位 | 同一策略名导出多次 → version +1 → magic 尾两位变 |
| **谱系版本** | lineage 链长度 | chain 越长 = 经过越多轮 tune 迭代 |

**spec_id 不变 = 因子内容完全相同 = 可以安全覆盖旧文件。** spec_id 变了 = 新因子 = 新文件。这是内容寻址的天然优势。

---

## 七、问题 4：完整生命周期

### 7.1 生命周期状态机

```
                              ┌────────────┐
                              │   CREATED   │  mine 产出 / tune 派生
                              └──────┬──────┘
                                     │ register
                              ┌──────▼──────┐
                              │   ACTIVE    │  默认状态
                              └──┬──┬──┬────┘
                  export         │  │  │  backtest
          ┌───────────────────────┘  │  └──────────────┐
          ▼                          │                  ▼
   ┌──────────────┐                  │ tune      ┌──────────────┐
   │  EXPORTED    │                  │           │  BACKTESTED  │
   │ (.py + magic) │                  │           │  (result)   │
   └──────────────┘                  ▼           └──────────────┘
                              ┌──────────────┐
                              │   TUNED      │
                              │ (child spec) │
                              └──────────────┘

   任何状态 ──archive──► ARCHIVED（不删文件，不参与默认列表）
```

### 7.2 各环节责任归属与数据流转

```
┌────────────┬──────────────┬───────────────────────────────┬──────────────────────┐
│  环节       │  触发者       │  数据流转                      │  Registry 操作        │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ mine       │ CLI / Web UI  │ 行情→搜索→门禁→Spec JSON       │ register(derivation=  │
│            │               │ → 写 spec.json + history.json  │   "mine")             │
│            │               │ → (可选) auto-export .py       → record_export()      │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ export     │ CLI / Web UI  │ 读 spec → compile → lint →     │ record_export()      │
│            │               │ 写 .py → 分配 magic             → magic 账本加 spec_id │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ verify     │ CLI           │ 读 .py → lint + 数值保真度回归  │ (只读，不写)           │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ backtest   │ CLI / Web UI  │ 读 spec → StackVM 求值 →        │ record_backtest()    │
│            │               │ 写 backtest_*.json              │                      │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ tune       │ CLI / Web UI  │ 读 spec → TPE 寻优 →           │ register(derivation=  │
│            │               │ TuneResult JSON + (新增)       │   "tune",parent_id=) │
│            │               │ 派生 spec JSON                  → record_tune(         │
│            │               │                                │   child_spec_id=)    │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ report     │ CLI / Web UI  │ 读 spec + registry →           │ (只读)               │
│            │               │ 打印谱系 + 导出历史 + 回测成绩  │                      │
├────────────┼──────────────┼───────────────────────────────┼──────────────────────┤
│ archive    │ CLI / Web UI  │ registry.archive(spec_id)      │ status → ARCHIVED    │
└────────────┴──────────────┴───────────────────────────────┴──────────────────────┘
```

### 7.3 CLI 命令增强

#### `mine` 命令

```bash
# 现有
miaosuan mine --data XAUUSD.csv --out artifacts/spec.json

# 增强：mine 完成后自动注册到 registry，可选自动导出
miaosuan mine --data XAUUSD.csv --out artifacts/spec.json \
    --auto-export                    # mine 完成后自动导出 DEPLOYABLE 的 spec
```

#### `export` 命令

```bash
# 现有
miaosuan export --spec artifacts/spec.json --out-dir shenji-strategies

# 增强：导出后自动更新 registry + magic 账本加 spec_id
miaosuan export --spec artifacts/spec.json --out-dir shenji-strategies \
    --force                          # 强制重新导出（覆盖幂等）
```

#### `tune` 命令

```bash
# 现有
miaosuan tune --spec artifacts/spec.json --data XAUUSD.csv

# 增强：tune 完成后产出派生 spec + 注册到 registry
miaosuan tune --spec artifacts/spec.json --data XAUUSD.csv \
    --derive-spec                    # 用最优参数产出派生 spec JSON
    --auto-export                    # 派生 spec 如过门禁则自动导出
```

#### 新增 `lineage` 命令

```bash
# 查看某个 spec 的完整谱系
miaosuan lineage --spec artifacts/spec.json

# 输出示例：
# spec 5d0841b6db9fdca2 (h1_xauusd_miaosuan) — BLOCKED
#   └── spec a3f8c2d1e9b70451 (h1_xauusd_miaosuan_tuned) — RESEARCH_ONLY
#         └── spec b7e1d9f3a2c8e504 (h1_xauusd_miaosuan_tuned_v2) — DEPLOYABLE ✓
#               exports: 20260925_h1_xauusd_miaosuan_tuned_v2.py (magic=662101)
```

### 7.4 report 命令增强

现有 `report` 只打印 evidence + provenance。增强为从 registry 拉取完整视图：

```
═══ Spec Report ═══
ID:          5d0841b6db9fdca2
Name:        h1_xauusd_miaosuan
Gate:        BLOCKED (dsr<0.5, dsr=0.036)
Created:     2026-09-16 04:58 UTC
Market:      FOREX_XAUUSD
Budget:      quick

── Lineage ──
  (root) ← mine, no parent
  ├── spec a3f8c2d1 (tuned, RESEARCH_ONLY)
  │     └── spec b7e1d9f3 (tuned, DEPLOYABLE) ← export → .py (magic=662101)

── Exports (1) ──
  20260913_h1_testxau_miaosuan_v1.py  magic=661801  DEPLOYABLE=False  lint=OK

── Backtests (2) ──
  2026-09-20  sharpe=4.70  mdd=0.12
  2026-09-21  sharpe=3.85  mdd=0.15

── Tunes (1) ──
  2026-09-21  best_score=2.1  → child: a3f8c2d1 (RESEARCH_ONLY)
```

---

## 八、向后兼容策略

| 变更 | 兼容性 | 说明 |
|------|--------|------|
| Provenance 新增 3 字段 | ✅ 完全兼容 | 旧 JSON 缺这些字段时 `from_dict` 用默认值 |
| SpecRegistry 新增 | ✅ 纯增量 | 不影响无 registry 的现有流程 |
| ExportResult 新增 spec_id | ✅ 可选字段 | 默认 None，不影响现有调用方 |
| magic 账本新增 spec_id | ✅ 可选字段 | 旧记录缺 spec_id 时按 None 处理 |
| .py 新增 SPEC_ID 常量 | ✅ 纯注释级 | 不影响 .py 的运行时行为 |
| tune 产出派生 spec | ✅ 可选功能 | `--derive-spec` flag，不加则行为不变 |
| `lineage` 命令 | ✅ 纯新增 | 不影响现有命令 |
| `rebuild_from_disk` | ✅ 灾难恢复 | 扫描 artifacts/ 重建 registry |

---

## 九、实施路线（按优先级排序）

### Phase 1：SpecRegistry 基础设施（最小可用）

1. 新建 `src/miaosuan/ir/registry.py`：`SpecRegistry` + `SpecRecord` 数据类
2. `mine` 命令结束时调用 `registry.register()`
3. `export` 命令结束时调用 `registry.record_export()`
4. magic 账本 record 追加 `spec_id` 字段
5. 模板新增 `SPEC_ID` 常量

### Phase 2：谱系追踪

1. `Provenance` 新增 `parent_spec_id` / `lineage` / `derivation` 字段
2. `tune` 命令新增 `--derive-spec` 选项，产出派生 spec
3. `build_tuned_spec()` 实现
4. 新增 `lineage` CLI 命令

### Phase 3：回测/寻优记录与 report 增强

1. `backtest` 命令结束时调用 `registry.record_backtest()`
2. `tune` 命令结束时调用 `registry.record_tune()`
3. `report` 命令增强为从 registry 拉取完整视图
4. Web UI 的 `lineageKv` div 接入真实谱系数据

### Phase 4：导出策略与自动化

1. `ExportPolicy` 配置类 + `should_export()` 决策函数
2. `mine --auto-export` 和 `tune --auto-export` flag
3. `export --force` flag
4. `archive` / `rebuild_from_disk` 命令

---

## 十、关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| registry 独立 vs 嵌入 spec | 独立 sidecar JSON | spec 的四段结构是"因子本身"的描述，registry 是"管理元数据"，混入会破坏 spec 的纯净性和确定性序列化 |
| Provenance 加字段 vs 独立 Lineage 对象 | 加字段（可选） | 向后兼容、序列化简单、frozen 不变 |
| tune 产出派生 spec vs 只存参数 | 产出派生 spec | 参数组合无法直接 export/backtest；派生 spec 才能形成完整闭环 |
| spec_id 用内容寻址 vs UUID | 内容寻址（已有） | 同因子重跑得同 ID，天然去重；UUID 无法去重 |
| .py 内嵌 SPEC_ID 常量 vs 只靠 registry | 两者都有 | 常量让 .py 文件自描述（离线可查）；registry 做索引和反查 |
| 导出决策用 policy 对象 vs 硬编码 | policy 对象 | 可配置、可测试、不同环境不同策略 |
