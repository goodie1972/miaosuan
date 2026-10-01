
"use strict";

/* ── 全局状态 ─────────────────────────────────────────────────────── */
var $ = function (id) { return document.getElementById(id); };
var state = {
  view: "mine",
  spec: null,
  logOffset: 0,
  polling: null,
  autoscroll: true,
  debug: false,
  cmd: "",
  candidates: [],
  errors: [],
  trace: []
};
var THEME_KEY = "miaosuan-theme";
var LANG_KEY = "miaosuan-lang";

/* ── i18n 翻译字典 ─────────────────────────────────────────────── */
var TRANSLATIONS = {
  zh: {
    title: "妙算 · 因子挖掘中心",
    subtitle: "MiaoSuan · 声明式因子挖掘 → 妙算 策略导出",
    statusIdle: "空闲",
    statusMining: "挖掘中",
    cfgMeta: "词表 — · tokens —",
    btnTheme: "◐ 主题",
    btnLang: "EN",
    navMine: "挖掘",
    navBacktest: "回测",
    navLive: "实时",
    navTune: "调优",
    step1Data: "① 数据文件",
    step1Hint: "— 指纹把「这次用了哪份数据」钉死",
    labelData: "行情数据（parquet / csv）",
    btnInspect: "读取元信息",
    dataMetaDefault: "选择数据文件后点「读取元信息」—— 展示品种 / 周期 / bar 数 / 年限 / 路径 / 数据指纹。",
    step2Title: "② 启动挖掘（mine）",
    labelBudget: "预算档位",
    budgetQuick: "quick · 约30分钟",
    budgetStd: "standard · 约2小时",
    budgetDeep: "deep · 约8小时",
    labelSeed: "种子（0=默认）",
    labelSymbol: "标的",
    labelTimeframe: "周期",
    labelMarket: "市场画像（留空 = 按标的自动推断）",
    btnStartMine: "开始挖掘",
    step3Title: "③ 进度与分数",
    kStatus: "状态",
    kReason: "停止原因",
    kGen: "代数",
    kEval: "评估次数",
    kVal: "最优 val_score",
    kDsr: "DSR（去膨胀夏普）",
    kDiv: "多样性",
    kGate: "门禁结论",
    step4Title: "④ 训练曲线与候选分数",
    step4Hint: "— 真实数据，非模拟",
    curveEmpty: "暂无逐代历史 —— 完成一次挖掘后，这里显示真实的训练曲线",
    curveHint: "数据来源与口径：<b>CLI 子进程输出的 Top-5 候选行</b>（<span class=\"mono\">[i] 公式 val=… dsr=… verdict=…</span>）。",
    step5Title: "⑤ 实时日志",
    step5Hint: "— 等宽字体 · 自动滚到底",
    labelCmd: "实际执行的 CLI 命令",
    btnScroll: "暂停滚动",
    logEmpty: "尚未启动任务。",
    step6Title: "⑥ 最优公式",
    formulaDefault: "（尚无结果）",
    btnCopy: "复制公式",
    step7Title: "⑦ 门禁与谱系",
    step7Hint: "— 门禁结论与溯源信息",
    lGate: "门禁结论",
    lReasons: "触发原因",
    lGit: "git sha",
    lSeed: "随机种子",
    lVocab: "词表版本（冻结锁）",
    lMarket: "市场画像",
    lFp: "数据指纹",
    btnVerify: "校验",
    step8Title: "⑧ 历史 Spec",
    step9Title: "⑨ 已导出策略（妙算 .py）",
    loading: "加载中…",
    noSpec: "（还没有 spec —— 先跑一次挖掘）",
    noStrategy: "（还没有导出策略）",
    step10Title: "⑩ 说明 / 帮助",
    step11Title: "⑪ 调试面板",
    btnDebug: "显示调试信息",
    debugHint: "默认关闭；打开后展示最近的前端异常与后端响应。",
    // 回测页
    btConfigTitle: "① 回测配置",
    btConfigHint: "— 子进程调 CLI，不直接 import 内核",
    labelSpec: "StrategySpec",
    labelBtData: "行情数据",
    labelBtMarket: "市场画像（可空）",
    labelBtWindow: "滚动窗口（0=自动）",
    btnBacktest: "运行回测",
    btRunning: "回测中…（子进程调 CLI，全样本逐 bar 计算）",
    btComplete: "完成 · 结果已落盘",
    kpiTitle: "② 绩效概览",
    kpiHint: "— 真实指标，非 val_score",
    kTotalReturn: "累计收益",
    kAnnualReturn: "年化收益",
    kSharpe: "Sharpe",
    kSortino: "Sortino",
    kMaxDrawdown: "最大回撤（相对）",
    kTrades: "交易数",
    kWinRate: "胜率",
    kProfitFactor: "盈亏比（均盈/均亏）",
    chartTitle: "③ 资金曲线",
    chartHint: "— 内联 SVG",
    rollingTitle: "④ 滚动 Sharpe",
    rollingHint: "— 滑窗，窗口未满处留空",
    tradesTitle: "⑤ 交易明细",
    tradesHint: "— 方向翻转 = 新的一笔",
    noTrades: "本次回测没有产生任何交易（仓位全程未过中性带）。",
    // 实时页
    liveTitle: "03 实时",
    bannerConnecting: "正在连接后端…",
    bannerRefreshing: "首次刷新中",
    readonlyNote: "<b>只读接入</b>：仅对后端发 <span class=\"mono\">GET</span> 白名单（引擎状态 / 报价 / K 线 / 信号），",
    noWrite: "<b>绝不下单、绝不平仓、绝不启停引擎</b>；没有写操作。后端地址",
    autoRefresh: "每 2 秒自动刷新",
    engineTitle: "① 引擎状态",
    quoteTitle: "② 当前报价",
    candlesTitle: "③ K 线",
    signalsTitle: "④ 最近信号",
    // 调优页
    tuneTitle: "04 调优",
    tuneConfigTitle: "① 调优配置",
    tuneConfigHint: "— TPE 采样 + 回测评估",
    labelTuneSpec: "策略 Spec",
    labelTuneData: "行情数据",
    labelTuneTrials: "试验次数",
    labelTuneMetric: "优化目标",
    labelTuneSeed: "随机种子",
    btnStartTune: "开始调优",
    tuneRunning: "调优中…",
    tuneComplete: "完成 · 结果已落盘",
    tuneResultsTitle: "② 调优结果",
    kBestScore: "最优评分",
    kElapsed: "耗时",
    kTrials: "试验次数",
    paramTable: "参数名",
    paramValue: "参数值",
    scoreCol: "评分",
    // 数据获取
    acquisitionTitle: "数据获取",
    sourcesTitle: "数据来源",
    cachedTitle: "本地缓存",
    btnFetch: "获取数据",
    btnRefreshCache: "刷新缓存列表",
    sourceShenji: "Shenji DB",
    sourceNetwork: "Network API",
    available: "可用",
    unavailable: "不可用",
    noCache: "暂无缓存数据",
    symbolLabel: "品种",
    timeframeLabel: "周期",
    sinceLabel: "起始时间（Unix秒，留空=增量）",
    // 帮助
    helpBar1: "bar1 语义 —— 为什么策略不会重绘？",
    helpBar1Desc: "导出文件的 <span class=\"mono\">generate_signal</span> 只用<b>已收盘 K 线</b>（<span class=\"mono\">candles[:-1]</span>），",
    helpDsr: "DSR 是什么？",
    helpDsrDesc: "DSR = Deflated Sharpe Ratio（去膨胀夏普）。搜索过程试了几千个公式，最优结果天然带“选择偏差”；",
    helpBlocked: "BLOCKED 意味着什么？",
    helpBlockedDesc: "未通过实盘门禁：导出文件会被强制打上 <span class=\"mono\">RESEARCH_ONLY</span> 警示块、",
    helpVocab: "词表冻结锁（VOCAB_VERSION）的作用",
    helpVocabDesc: "token 只在<b>同一词表</b>下有意义。导出文件内嵌 <span class=\"mono\">VOCAB_VERSION</span>，",
    // Spec 管理
    navSpecs: "Spec 管理",
    spmOverviewTitle: "① 概览",
    spmOverviewHint: "— Spec 全景统计",
    spmListTitle: "② Spec 列表",
    spmListHint: "— 点击查看详情 / 谱系 / 关联",
    spmDetailTitle: "③ Spec 详情",
    spmLineageTitle: "④ 谱系",
    spmLineageHint: "— 派生关系链",
    spmExportsTitle: "⑤ 导出记录",
    spmExportsHint: "— 该 Spec 产出的所有 .py 策略",
    spmBacktestsTitle: "⑥ 回测记录",
    spmBacktestsHint: "— 该 Spec 的所有回测结果",
    spmTunesTitle: "⑦ 寻优记录",
    spmTunesHint: "— 该 Spec 的所有调优试验",
    spmTotalSpecs: "Spec 总数",
    spmTotalStrategies: "导出策略数",
    spmTotalBacktests: "回测次数",
    spmTotalTunes: "寻优次数",
    spmSearchPlaceholder: "输入关键字过滤…",
    spmNoExports: "无导出记录",
    spmNoBacktests: "无回测记录",
    spmNoTunes: "无寻优记录",
    spmLineageRoot: "当前 Spec 为根节点（mine 产出，无父 Spec）",
    spmPayload: "Payload（因子体）",
    spmSemantics: "Semantics（语义）",
    spmEvidence: "Evidence（证据）",
    spmProvenance: "Provenance（溯源）",
    // 设置页
    navConfig: "设置",
    cfgTitle: "① 系统配置",
    cfgHint: "— 修改后写入 settings.yaml 并即时热重载，无需重启",
    cfgSectionWebUI: "WebUI 服务",
    cfgSectionShenji: "神机仪表盘",
    cfgSectionMT4: "MT4 Bridge",
    cfgSectionPaths: "路径设置",
    cfgSectionData: "数据源",
    cfgSectionLogging: "日志",
    cfgSectionFeatures: "功能开关",
    cfgPort: "端口",
    cfgHost: "主机",
    cfgReload: "热重载(开发)",
    cfgTimeout: "超时(秒)",
    cfgTimeBase: "时间基准",
    cfgPollWait: "轮询间隔(ms)",
    cfgDrainWait: "排空等待(ms)",
    cfgDataCache: "数据缓存目录",
    cfgArtifacts: "产物目录",
    cfgTmp: "临时目录",
    cfgKline: "K线数据根目录",
    cfgShenjiDB: "神机数据库路径",
    cfgDataTimeout: "网络超时(秒)",
    cfgDataSource: "默认数据源",
    cfgDataAPIUrl: "HTTP数据源地址",
    cfgDataCacheDir: "数据缓存目录(留空用paths)",
    cfgDukascopyUser: "Dukascopy用户名",
    cfgDukascopyPass: "Dukascopy密码",
    cfgLogLevel: "日志级别",
    cfgLogJSON: "JSON格式",
    cfgEnableWebUI: "启用 WebUI",
    cfgEnableMT4: "启用 MT4",
    cfgEnableShenji: "启用神机",
    cfgEnableSpec: "启用 Spec 管理",
    cfgEnableBacktest: "启用回测",
    cfgEnableTune: "启用调优",
    btnSaveConfig: "保存配置",
    btnReloadConfig: "重新加载",
    cfgSaving: "保存中…",
    cfgSaved: "已保存并热重载",
    cfgError: "保存失败：",
    cfgCurrentFile: "当前配置文件：",
    cfgEnvNote: "注：环境变量 MIAOSUAN_* 优先级高于此文件，修改环境变量后需重启进程生效。",
    cfgRequiresRestart: "⚠ 修改此项需重启进程生效",
  },
  en: {
    title: "MiaoSuan · Factor Mining Center",
    subtitle: "Declarative Factor Mining → Shenji Strategy Export",
    statusIdle: "Idle",
    statusMining: "Mining...",
    cfgMeta: "Vocab — · tokens —",
    btnTheme: "◐ Theme",
    btnLang: "中文",
    navMine: "Mine",
    navBacktest: "Backtest",
    navLive: "Live",
    navTune: "Tune",
    step1Data: "① Data File",
    step1Hint: "— Fingerprint locks data provenance",
    labelData: "Market Data (parquet / csv)",
    btnInspect: "Inspect",
    dataMetaDefault: "Select a data file and click Inspect to show symbol/timeframe/bars/duration/path/fingerprint.",
    step2Title: "② Start Mining (mine)",
    labelBudget: "Budget",
    budgetQuick: "quick ~30min",
    budgetStd: "standard ~2h",
    budgetDeep: "deep ~8h",
    labelSeed: "Seed (0=default)",
    labelSymbol: "Symbol",
    labelTimeframe: "Timeframe",
    labelMarket: "Market Profile (auto if empty)",
    btnStartMine: "Start Mining",
    step3Title: "③ Progress & Score",
    kStatus: "Status",
    kReason: "Stop Reason",
    kGen: "Generations",
    kEval: "Evaluations",
    kVal: "Best Val-Score",
    kDsr: "DSR (Deflated Sharpe)",
    kDiv: "Diversity",
    kGate: "Gate Verdict",
    step4Title: "④ Training Curve & Candidates",
    step4Hint: "— Real data, not simulated",
    curveEmpty: "No history yet — complete a mining run to see the training curve.",
    curveHint: "Source: <b>CLI Top-5 candidate lines</b> (<span class=\"mono\">[i] formula val=… dsr=… verdict=…</span>).",
    step5Title: "⑤ Live Log",
    step5Hint: "— Monospace · Auto-scroll",
    labelCmd: "Actual CLI Command",
    btnScroll: "Pause Scroll",
    logEmpty: "No task started yet.",
    step6Title: "⑥ Best Formula",
    formulaDefault: "(No result yet)",
    btnCopy: "Copy Formula",
    step7Title: "⑦ Gate & Provenance",
    step7Hint: "— Gate verdict & provenance",
    lGate: "Gate Verdict",
    lReasons: "Trigger Reasons",
    lGit: "Git SHA",
    lSeed: "Random Seed",
    lVocab: "Vocab Version (Frozen Lock)",
    lMarket: "Market Profile",
    lFp: "Data Fingerprint",
    btnVerify: "Verify",
    step8Title: "⑧ History Specs",
    step9Title: "⑨ Exported Strategies (Shenji .py)",
    loading: "Loading…",
    noSpec: "(No spec yet — run mining first)",
    noStrategy: "(No exported strategies yet)",
    step10Title: "⑩ Help / Documentation",
    step11Title: "⑪ Debug Panel",
    btnDebug: "Show Debug Info",
    debugHint: "Off by default; opens recent frontend errors and backend responses.",
    // Backtest
    btConfigTitle: "① Backtest Config",
    btConfigHint: "— Subprocess calls CLI, no direct kernel import",
    labelSpec: "StrategySpec",
    labelBtData: "Market Data",
    labelBtMarket: "Market Profile (optional)",
    labelBtWindow: "Rolling Window (0=auto)",
    btnBacktest: "Run Backtest",
    btRunning: "Backtesting… (subprocess, full-sample per-bar)",
    btComplete: "Done · Result saved to",
    kpiTitle: "② Performance Summary",
    kpiHint: "— Real metrics, not val_score",
    kTotalReturn: "Total Return",
    kAnnualReturn: "Annual Return",
    kSharpe: "Sharpe",
    kSortino: "Sortino",
    kMaxDrawdown: "Max Drawdown (rel)",
    kTrades: "Trades",
    kWinRate: "Win Rate",
    kProfitFactor: "Profit Factor",
    chartTitle: "③ Equity Curve",
    chartHint: "— Inline SVG",
    rollingTitle: "④ Rolling Sharpe",
    rollingHint: "— Sliding window, empty where insufficient",
    tradesTitle: "⑤ Trade Details",
    tradesHint: "— Direction flip = new trade",
    noTrades: "No trades generated (position never exceeded neutral band).",
    // Live
    liveTitle: "03 Live",
    bannerConnecting: "Connecting to backend…",
    bannerRefreshing: "First refresh in progress",
    readonlyNote: "<b>Read-only access</b>: only <span class=\"mono\">GET</span> whitelisted endpoints (engine status / quotes / candles / signals),",
    noWrite: "<b>No orders, no closes, no engine control</b>. Backend address",
    autoRefresh: "Auto-refresh every 2s",
    engineTitle: "① Engine Status",
    quoteTitle: "② Current Quote",
    candlesTitle: "③ Candles",
    signalsTitle: "④ Latest Signals",
    // Tune
    tuneTitle: "04 Tune",
    tuneConfigTitle: "① Tuning Config",
    tuneConfigHint: "— TPE sampling + backtest evaluation",
    labelTuneSpec: "Strategy Spec",
    labelTuneData: "Market Data",
    labelTuneTrials: "Trial Count",
    labelTuneMetric: "Optimization Metric",
    labelTuneSeed: "Random Seed",
    btnStartTune: "Start Tuning",
    tuneRunning: "Tuning in progress…",
    tuneComplete: "Done · Result saved",
    tuneResultsTitle: "② Tuning Results",
    kBestScore: "Best Score",
    kElapsed: "Elapsed",
    kTrials: "Trials",
    paramTable: "Parameter",
    paramValue: "Value",
    scoreCol: "Score",
    // Acquisition
    acquisitionTitle: "Data Acquisition",
    sourcesTitle: "Data Sources",
    cachedTitle: "Local Cache",
    btnFetch: "Fetch Data",
    btnRefreshCache: "Refresh Cache List",
    sourceShenji: "Shenji DB",
    sourceNetwork: "Network API",
    available: "Available",
    unavailable: "Unavailable",
    noCache: "No cached data",
    symbolLabel: "Symbol",
    timeframeLabel: "Timeframe",
    sinceLabel: "Since (Unix ts, empty=incremental)",
    // Help
    helpBar1: "bar1 semantics — why no repaint?",
    helpBar1Desc: "The exported <span class=\"mono\">generate_signal</span> uses only <b>closed candles</b> (<span class=\"mono\">candles[:-1]</span>),",
    helpDsr: "What is DSR?",
    helpDsrDesc: "DSR = Deflated Sharpe Ratio. The search tests thousands of formulas; optimal results have selection bias.",
    helpBlocked: "What does BLOCKED mean?",
    helpBlockedDesc: "Failed live gate: exported file gets <span class=\"mono\">RESEARCH_ONLY</span> tag,",
    helpVocab: "Vocab frozen lock (VOCAB_VERSION)",
    helpVocabDesc: "Tokens only make sense under the <b>same vocabulary</b>. Exported files embed <span class=\"mono\">VOCAB_VERSION</span>,",
    // Spec Management
    navSpecs: "Specs",
    spmOverviewTitle: "① Overview",
    spmOverviewHint: "— Spec panorama statistics",
    spmListTitle: "② Spec List",
    spmListHint: "— Click for details / lineage / relations",
    spmDetailTitle: "③ Spec Detail",
    spmLineageTitle: "④ Lineage",
    spmLineageHint: "— Derivation chain",
    spmExportsTitle: "⑤ Export Records",
    spmExportsHint: "— All .py strategies from this spec",
    spmBacktestsTitle: "⑥ Backtest Records",
    spmBacktestsHint: "— All backtests for this spec",
    spmTunesTitle: "⑦ Tune Records",
    spmTunesHint: "— All tuning trials for this spec",
    spmTotalSpecs: "Total Specs",
    spmTotalStrategies: "Exported Strategies",
    spmTotalBacktests: "Backtests",
    spmTotalTunes: "Tunes",
    spmSearchPlaceholder: "Filter by keyword…",
    spmNoExports: "No exports",
    spmNoBacktests: "No backtests",
    spmNoTunes: "No tunes",
    spmLineageRoot: "Root node (mine output, no parent)",
    spmPayload: "Payload (Factor)",
    spmSemantics: "Semantics",
    spmEvidence: "Evidence",
    spmProvenance: "Provenance",
    // Config page
    navConfig: "Config",
    cfgTitle: "① System Config",
    cfgHint: "— Changes written to settings.yaml and hot-reloaded instantly, no restart needed",
    cfgSectionWebUI: "WebUI Service",
    cfgSectionShenji: "Shenji Dashboard",
    cfgSectionMT4: "MT4 Bridge",
    cfgSectionPaths: "Paths",
    cfgSectionData: "Data Sources",
    cfgSectionLogging: "Logging",
    cfgSectionFeatures: "Feature Flags",
    cfgPort: "Port",
    cfgHost: "Host",
    cfgReload: "Hot Reload (dev)",
    cfgTimeout: "Timeout (s)",
    cfgTimeBase: "Time Base",
    cfgPollWait: "Poll Wait (ms)",
    cfgDrainWait: "Drain Wait (ms)",
    cfgDataCache: "Data Cache Dir",
    cfgArtifacts: "Artifacts Dir",
    cfgTmp: "Temp Dir",
    cfgKline: "K-line Root Dir",
    cfgShenjiDB: "Shenji DB Path",
    cfgDataTimeout: "Network Timeout (s)",
    cfgDataSource: "Default Source",
    cfgDataAPIUrl: "HTTP Source URL",
    cfgDataCacheDir: "Data Cache Dir (empty=paths)",
    cfgDukascopyUser: "Dukascopy User",
    cfgDukascopyPass: "Dukascopy Pass",
    cfgLogLevel: "Log Level",
    cfgLogJSON: "JSON Format",
    cfgEnableWebUI: "Enable WebUI",
    cfgEnableMT4: "Enable MT4",
    cfgEnableShenji: "Enable Shenji",
    cfgEnableSpec: "Enable Spec Mgmt",
    cfgEnableBacktest: "Enable Backtest",
    cfgEnableTune: "Enable Tune",
    btnSaveConfig: "Save Config",
    btnReloadConfig: "Reload",
    cfgSaving: "Saving…",
    cfgSaved: "Saved & hot-reloaded",
    cfgError: "Save failed: ",
    cfgCurrentFile: "Config file: ",
    cfgEnvNote: "Note: Env vars MIAOSUAN_* override this file; change env vars and restart process to take effect.",
    cfgRequiresRestart: "⚠ Requires process restart",
  }
};

function t(key) {
  var lang = localStorage.getItem(LANG_KEY) || "zh";
  var dict = TRANSLATIONS[lang] || TRANSLATIONS.zh;
  return dict[key] || key;
}

function applyLanguage() {
  var lang = localStorage.getItem(LANG_KEY) || "zh";
  document.documentElement.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
  var btn = $("btnLang");
  if (btn) { btn.textContent = t("btnLang"); }
  var title = $("pageTitle");
  if (title) { title.textContent = t("title"); }
  var sub = $("pageSubtitle");
  if (sub) { sub.textContent = t("subtitle"); }
  var statusText = $("statusText");
  if (statusText && !state.polling) { statusText.textContent = t("statusIdle"); }
  var cfgMeta = $("cfgMeta");
  if (cfgMeta && cfgMeta.textContent.indexOf("词表") === -1 && cfgMeta.textContent.indexOf("Vocab") === -1) {
    cfgMeta.textContent = t("cfgMeta");
  }
  // Step headers
  var h2s = document.querySelectorAll("section.card > h2");
  h2s.forEach(function(h) {
    var txt = h.textContent || "";
    if (txt.indexOf("① 数据文件") !== -1) { h.childNodes[0].textContent = t("step1Data") + " "; }
    if (txt.indexOf("② 启动挖掘") !== -1) { h.childNodes[0].textContent = t("step2Title") + " "; }
    if (txt.indexOf("③ 进度") !== -1) { h.childNodes[0].textContent = t("step3Title") + " "; }
    if (txt.indexOf("④ 训练曲线") !== -1) { h.childNodes[0].textContent = t("step4Title") + " "; }
    if (txt.indexOf("⑤ 实时日志") !== -1) { h.childNodes[0].textContent = t("step5Title") + " "; }
    if (txt.indexOf("⑥ 最优公式") !== -1) { h.childNodes[0].textContent = t("step6Title") + " "; }
    if (txt.indexOf("⑦ 门禁") !== -1) { h.childNodes[0].textContent = t("step7Title") + " "; }
    if (txt.indexOf("⑧ 历史 Spec") !== -1) { h.childNodes[0].textContent = t("step8Title") + " "; }
    if (txt.indexOf("⑨ 已导出") !== -1) { h.childNodes[0].textContent = t("step9Title") + " "; }
    if (txt.indexOf("⑩ 说明") !== -1) { h.childNodes[0].textContent = t("step10Title") + " "; }
    if (txt.indexOf("⑪ 调试") !== -1) { h.childNodes[0].textContent = t("step11Title") + " "; }
  });
  // Nav buttons
  var navBtns = document.querySelectorAll("#stepper button");
  navBtns.forEach(function(btn) {
    var view = btn.getAttribute("data-view");
    if (view === "mine") btn.innerHTML = '<span class="step-no">01</span>' + t("navMine");
    if (view === "backtest") btn.innerHTML = '<span class="step-no">02</span>' + t("navBacktest");
    if (view === "live") btn.innerHTML = '<span class="step-no">03</span>' + t("navLive");
    if (view === "tune") btn.innerHTML = '<span class="step-no">04</span>' + t("navTune");
    if (view === "specs") btn.innerHTML = '<span class="step-no">05</span>' + t("navSpecs");
    if (view === "config") btn.innerHTML = '<span class="step-no">06</span>' + t("navConfig");
  });
  // KV labels
  var kvPairs = document.querySelectorAll(".kv .k");
  kvPairs.forEach(function(k) {
    var txt = k.textContent || "";
    if (txt === "状态") k.textContent = t("kStatus");
    if (txt === "停止原因") k.textContent = t("kReason");
    if (txt === "代数") k.textContent = t("kGen");
    if (txt === "评估次数") k.textContent = t("kEval");
    if (txt === "最优 val_score") k.textContent = t("kVal");
    if (txt === "DSR（去膨胀夏普）") k.textContent = t("kDsr");
    if (txt === "多样性") k.textContent = t("kDiv");
    if (txt === "门禁结论") k.textContent = t("kGate");
  });
  // Buttons
  var btnInspect = $("btnInspect");
  if (btnInspect) btnInspect.textContent = t("btnInspect");
  var btnMine = $("btnMine");
  if (btnMine) btnMine.textContent = t("btnStartMine");
  var btnScroll = $("btnScroll");
  if (btnScroll) btnScroll.textContent = state.autoscroll ? t("btnScroll") : "恢复滚动";
  var btnCopy = $("btnCopy");
  if (btnCopy) btnCopy.textContent = t("btnCopy");
  var btnVerify = $("btnVerify");
  if (btnVerify) btnVerify.textContent = t("btnVerify");
  var btnBacktest = $("btnBacktest");
  if (btnBacktest) btnBacktest.textContent = t("btnBacktest");
  var btnDebug = $("btnDebug");
  if (btnDebug) btnDebug.textContent = state.debug ? "隐藏调试信息" : t("btnDebug");
}

function switchLang() {
  var current = localStorage.getItem(LANG_KEY) || "zh";
  var next = current === "zh" ? "en" : "zh";
  localStorage.setItem(LANG_KEY, next);
  applyLanguage();
}

function esc(s) {
  return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function dbg(msg) {
  state.trace.push(new Date().toISOString() + "  " + msg);
  if (state.trace.length > 50) { state.trace.shift(); }
  if (state.debug) { $("dbgOut").textContent = state.trace.join("\n"); }
}

/* ── 主题（localStorage，切 data-theme 属性，不刷新）───────────────── */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  var logo = document.querySelector(".brand .mark img");
  if (logo) { logo.src = "/static/" + (t === "dark" ? "logo_dark.png" : "logo.png"); }
  try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* 隐私模式忽略 */ }
}
function initTheme() {
  var saved = "dark";
  try { saved = localStorage.getItem(THEME_KEY) || "dark"; } catch (e) { /* ignore */ }
  if (saved !== "dark" && saved !== "light") { saved = "dark"; }
  applyTheme(saved);
}

/* ── 视图切换 ─────────────────────────────────────────────────────── */
function setView(v) {
  state.view = v;
  ["mine", "backtest", "live", "tune", "specs", "config"].forEach(function (name) {
    var el = $("view-" + name);
    if (el) { el.className = "view" + (name === v ? " active" : ""); }
  });
  var btns = document.querySelectorAll("#stepper button");
  for (var i = 0; i < btns.length; i++) {
    var b = btns[i];
    if (b.getAttribute("data-view") === v) { b.className = "active"; } else { b.className = ""; }
  }
  // 03 实时页只在可见时轮询后端（不可见则停，避免空转干扰平台）。
  if (v === "live") { rtSetPolling($("rtPoll").checked); } else { rtSetPolling(false); }
  // 05 Spec 管理：进入时刷新数据
  if (v === "specs") { loadSpecMgmt(); }
  // 06 设置：进入时加载配置
  if (v === "config") { loadConfig(); }
}

/* ── 请求封装 ─────────────────────────────────────────────────────── */
function jfetch(url, opt) {
  return fetch(url, opt).then(function (r) {
    // 先容错解析 body：后端 4xx/5xx 也返回 JSON {"detail": ...}，非 JSON 时退化成 null。
    return r.json().catch(function () { return null; }).then(function (body) {
      if (!r.ok) {
        var detail = (body && body.detail) ? body.detail : ("HTTP " + r.status);
        throw new Error(detail);
      }
      dbg("OK  " + url);
      return body;
    });
  }).catch(function (err) {
    state.errors.push(url + " → " + err.message);
    dbg("ERR " + url + " → " + err.message);
    throw err;
  });
}
function post(url, payload) {
  return jfetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

/* ── 全局配置 meta ────────────────────────────────────────────────── */
function loadMeta() {
  jfetch("/api/meta").then(function (m) {
    $("cfgMeta").textContent = "词表 " + m.vocab_version
      + " · " + m.n_tokens + " tokens（" + m.n_features + " 特征 + " + m.n_operators + " 算子）"
      + " · v" + m.miaosuan_version;
    window.__profileDetails = m.profile_details || {};
    // 行情目录由后端下发（可用 MIAOSUAN_KLINE_DIR 覆盖），前端不再硬编码一份
    window.__dataDirs = m.data_dirs || [];
    var names = m.profiles || [];
    // 挖掘页画像（共享同一选项列表）
    var sel = $("market");
    var html = '<option value="" selected>自动推断</option>';
    names.forEach(function (p) { html += '<option value="' + esc(p) + '">' + esc(p) + "</option>"; });
    sel.innerHTML = html;
    // 数据获取页画像（独立选择状态，同一选项列表）
    var psel = $("acqProfile");
    if (psel) {
      var phtml = '<option value="" selected>请选择市场画像</option>';
      names.forEach(function (p) { phtml += '<option value="' + esc(p) + '">' + esc(p) + "</option>"; });
      psel.innerHTML = phtml;
    }
  }).catch(function () {
    $("cfgMeta").textContent = "词表 — · tokens —";
  });
}

/* ── 数据文件 ─────────────────────────────────────────────────────── */
/* 时间范围文案：取不到就显示 —（绝不露出 null / undefined） */
function dataRangeText(d) {
  return (d && d.start && d.end) ? (d.start + " → " + d.end) : "—";
}

/* 挖掘区那行「当前数据」提示：直接用 loadData 已拿到的 rows，不发新请求 */
function updateMineDataHint() {
  var el = $("mineDataHint");
  if (!el) return;
  var p = $("data").value;
  if (!p) { el.textContent = "当前数据：未选择数据文件"; return; }
  var rows = window.__dataRows || [];
  var hit = null;
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].path === p) { hit = rows[i]; break; }
  }
  var name = String(p).split(/[\\/]/).pop();
  el.textContent = "当前数据：" + name + " · " + dataRangeText(hit);
}

/* prefer：刷新后优先选中的路径（数据获取刚落盘的新文件用它）。
   否则保留刷新前的选中——挖完发现选中的文件没了，体验很差。 */
function loadData(prefer) {
  var prev = prefer || $("data").value;
  jfetch("/api/data").then(function (rows) {
    window.__dataRows = rows || [];
    if (!rows.length) {
      // 目录来自 /api/meta（后端按 MIAOSUAN_KLINE_DIR 解析），取不到时回落字面默认值
      var dirs = (window.__dataDirs && window.__dataDirs.length)
        ? window.__dataDirs
        : ["D:\\K线数据\\"];
      $("data").innerHTML = '<option value="">（未找到行情文件 —— 请将 parquet 放入 '
        + esc(dirs.join(" 或 ")) + '）</option>';
      updateMineDataHint();
      return;
    }
    $("data").innerHTML = rows.map(function (d) {
      return '<option value="' + esc(d.path) + '">' + esc(d.name) + " · "
        + esc(dataRangeText(d)) + " · " + d.size_mb + "MB · " + esc(d.mtime) + "</option>";
    }).join("");
    // 原选中文件若仍在列表中，重新选中（刷新不能清空用户选择）
    if (prev) {
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].path === prev) { $("data").value = prev; break; }
      }
    }
    updateMineDataHint();
  }).catch(function (e) {
    window.__dataRows = [];
    $("data").innerHTML = '<option value="">（行情列表加载失败：' + esc(e.message) + "）</option>";
    updateMineDataHint();
  });
}

function inspect() {
  var p = $("data").value;
  if (!p) { $("dataMeta").className = "empty"; $("dataMeta").textContent = "请先选择数据文件。"; return; }
  $("dataMeta").className = "empty";
  $("dataMeta").textContent = "读取中…";
  jfetch("/api/inspect?path=" + encodeURIComponent(p)).then(function (m) {
    $("dataMeta").className = "kv";
    $("dataMeta").innerHTML = [
      ["品种", m.symbol], ["周期", m.timeframe], ["bar 数", m.n_bars],
      ["数据年限", m.years + " 年"], ["市场画像", m.market_profile || "(数据未自带)"],
      ["时间范围", m.start + " → " + m.end]
    ].map(function (kv) {
      return '<div><div class="k">' + esc(kv[0]) + '</div><div class="v">' + esc(kv[1]) + "</div></div>";
    }).join("") +
      '<div style="grid-column:1/-1"><div class="k">文件路径</div><div class="v">' + esc(m.path) + "</div></div>" +
      '<div style="grid-column:1/-1"><div class="k">数据指纹</div><div class="v">' + esc(m.fingerprint) + "</div></div>";
  }).catch(function (e) {
    $("dataMeta").className = "empty";
    $("dataMeta").textContent = "读取失败：" + e.message;
  });
}

/* 自动回填周期/品种：选中数据文件后自动调用 /api/inspect，把 timeframe/symbol 写入输入框 */
function autoFillTimeframe() {
  var p = $("data").value;
  if (!p) return;
  jfetch("/api/inspect?path=" + encodeURIComponent(p)).then(function (m) {
    var tf = m.timeframe;
    var sym = m.symbol;
    if (tf && !$("timeframe").value) { $("timeframe").value = tf; }
    if (sym && !$("symbol").value) { $("symbol").value = sym.split(",")[0].trim(); }
  }).catch(function () { /* 静默失败，不影响主流程 */ });
}

/* 回测页：数据切换时自动回填周期到 btTimeframe（若存在） */
function autoFillTimeframeForBt() {
  var p = $("btData").value;
  if (!p) return;
  jfetch("/api/inspect?path=" + encodeURIComponent(p)).then(function (m) {
    var tf = m.timeframe;
    var el = $("btTimeframe");
    if (tf && el && !el.value) { el.value = tf; }
  }).catch(function () { /* 静默失败 */ });
}

/* 调优页：数据切换时自动回填周期到 tuneTimeframe（若存在） */
function autoFillTimeframeForTune() {
  var p = $("tuneData").value;
  if (!p) return;
  jfetch("/api/inspect?path=" + encodeURIComponent(p)).then(function (m) {
    var tf = m.timeframe;
    var el = $("tuneTimeframe");
    if (tf && el && !el.value) { el.value = tf; }
  }).catch(function () { /* 静默失败 */ });
}

/* ── 日志解析（全部来自 CLI 真实输出，不做任何模拟）────────────────── */
function parseLog(lines) {
  var stats = { reason: "—", gen: "—", evalN: "—", div: "—", holdout: "" };
  var cands = [];
  lines.forEach(function (line) {
    var m;
    if ((m = line.match(/停止原因：(\S+)\s+代数=(\d+)\s+评估=(\d+)/))) {
      stats.reason = m[1]; stats.gen = m[2]; stats.evalN = m[3];
    }
    if ((m = line.match(/多样性：最低\s+([\d.]+)\s*\/\s*初始\s+([\d.]+)\s*=\s*([\d.]+)/))) {
      stats.div = m[3] + " (" + m[1] + "/" + m[2] + ")";
    }
    if ((m = line.match(/开发区\s+(\d+)\s*\/\s*全量\s+(\d+)\s*根（hold-out 未触碰=(\S+?)）?$/))) {
      stats.holdout = "开发区 " + m[1] + " / 全量 " + m[2] + " 根 · hold-out 未触碰=" + (m[3] || "—");
    }
    if ((m = line.match(/^\[(\d)\]\s+(.+?)\s+val=([\d.]+)\s+dsr=([\d.eE+-]+)\s+verdict=(\S+)/))) {
      cands.push({ i: parseInt(m[1], 10), formula: m[2], val: parseFloat(m[3]), dsr: parseFloat(m[4]), verdict: m[5] });
    }
  });
  return { stats: stats, candidates: cands };
}

/* ── 训练曲线（逐代最优/平均，内联 SVG 手绘，零依赖）────────────────── */
/* 数据源：CLI 写的 sidecar artifacts/<spec stem>.history.json（逐代 GAStats）。
   不插值、不平滑：缺哪代就是哪代没产出，宁可断线也不脑补。            */
function renderCurve(points) {
  var box = $("curve");
  var note = $("curveNote");
  if (!points || !points.length) {
    // 优雅降级：历史为空（本次未产出 / 还没跑）就明确说清楚，绝不画空图或报错。
    box.innerHTML = '<div class="empty">暂无逐代历史 —— 完成一次挖掘后，这里显示真实的训练曲线'
      + '（数据来源：CLI 写出的 <span class="mono">.history.json</span> sidecar）。</div>';
    note.textContent = "";
    return;
  }
  var W = 640, H = 230, padL = 58, padR = 16, padT = 16, padB = 34;
  var innerW = W - padL - padR, innerH = H - padT - padB;
  var lo = Infinity, hi = -Infinity, gLo = Infinity, gHi = -Infinity;
  points.forEach(function (p) {
    if (p.best < lo) { lo = p.best; }
    if (p.mean < lo) { lo = p.mean; }
    if (p.best > hi) { hi = p.best; }
    if (p.mean > hi) { hi = p.mean; }
    if (p.generation < gLo) { gLo = p.generation; }
    if (p.generation > gHi) { gHi = p.generation; }
  });
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) { hi = lo + 1; }          // 全平：避免除以 0
  var pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  if (gHi === gLo) { gHi = gLo + 1; }            // 只有一代：避免除以 0

  function xOf(g) { return padL + (g - gLo) / (gHi - gLo) * innerW; }
  function yOf(v) { return padT + (hi - v) / (hi - lo) * innerH; }

  function pathOf(key, color) {
    var d = "";
    points.forEach(function (p, k) {
      d += (k ? " L" : "M") + xOf(p.generation).toFixed(2) + " " + yOf(p[key]).toFixed(2);
    });
    return '<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="2"'
      + ' stroke-linejoin="round" stroke-linecap="round" style="color:' + color + '"></path>';
  }

  var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" height="' + H + '"'
    + ' role="img" aria-label="逐代最优分与平均分曲线">';
  var mono = 'font-family:var(--mono);font-size:10px';
  // 网格 + Y 轴刻度（4 档）
  for (var i = 0; i <= 4; i++) {
    var v = lo + (hi - lo) * i / 4, y = yOf(v);
    svg += '<line x1="' + padL + '" y1="' + y.toFixed(2) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(2)
      + '" style="stroke:var(--line);stroke-width:1"></line>';
    svg += '<text x="' + (padL - 6) + '" y="' + (y + 3).toFixed(2) + '" text-anchor="end"'
      + ' style="fill:var(--ink-2);' + mono + '">' + v.toFixed(3) + "</text>";
  }
  // X 轴刻度（首 / 中 / 末代）
  [gLo, Math.round((gLo + gHi) / 2), gHi].forEach(function (g) {
    svg += '<text x="' + xOf(g).toFixed(2) + '" y="' + (H - padB + 16) + '" text-anchor="middle"'
      + ' style="fill:var(--ink-2);' + mono + '">gen ' + g + "</text>";
  });
  svg += pathOf("mean", "var(--info)");
  svg += pathOf("best", "var(--gold)");
  // 末代打点，便于读终值
  var last = points[points.length - 1];
  svg += '<circle cx="' + xOf(last.generation).toFixed(2) + '" cy="' + yOf(last.best).toFixed(2)
    + '" r="3" style="fill:var(--gold)"></circle>';
  svg += '<circle cx="' + xOf(last.generation).toFixed(2) + '" cy="' + yOf(last.mean).toFixed(2)
    + '" r="3" style="fill:var(--info)"></circle>';
  svg += "</svg>";
  box.innerHTML = svg;

  var legend = '<span style="color:var(--gold)">━ 最优分</span> &nbsp;'
    + '<span style="color:var(--info)">━ 平均分</span> &nbsp; '
    + points.length + " 代 &nbsp;·&nbsp; 终值 最优 " + last.best.toFixed(4)
    + " / 平均 " + last.mean.toFixed(4);
  note.innerHTML = "训练曲线口径：" + legend
    + "。<b>适配度 = GA 搜索适应度</b>（非绩效指标，不能直接当收益/夏普读）；"
    + "横轴为真实代数，纵轴为该代适应度，<b>未做插值与平滑</b>。";
}

/* ── Top-5 候选柱状图（内联 SVG 手绘，零依赖）─────────────────────── */
function verdictColor(v) {
  if (v === "DEPLOYABLE" || v === "PASS") { return "var(--up)"; }
  if (v === "BLOCKED" || v === "FAIL") { return "var(--down)"; }
  return "var(--warn)";
}
function renderChart(cands) {
  var box = $("chart");
  if (!cands || !cands.length) {
    box.innerHTML = '<div class="empty">暂无候选数据 —— 完成一次挖掘后，这里显示 CLI 输出的 Top-5 候选真实得分。</div>';
    return;
  }
  var max = 0;
  cands.forEach(function (c) { if (c.val > max) { max = c.val; } });
  if (max <= 0) { max = 1; }
  var rowH = 32, x0 = 210, barW = 360, h = cands.length * rowH + 14;
  var svg = '<svg viewBox="0 0 640 ' + h + '" width="100%" height="' + h + '" role="img" aria-label="Top 候选 val_score">';
  // 颜色走 style 而不是 fill= 属性：表现属性里用 var() 在部分浏览器不代换，
  // 会整块变黑（或不可见）。style 里用 var() 是稳定支持的写法。
  var mono = 'font-family:var(--mono);font-size:11px';
  cands.forEach(function (c, k) {
    var y = k * rowH + 6;
    var w = Math.max(2, Math.round(c.val / max * barW));
    var label = c.formula.length > 26 ? c.formula.slice(0, 25) + "…" : c.formula;
    svg += '<text x="4" y="' + (y + 15) + '" style="fill:var(--ink-2);' + mono + '">' + esc(label) + "</text>";
    svg += '<rect x="' + x0 + '" y="' + (y + 3) + '" width="' + w + '" height="16" rx="3"'
      + ' style="fill:' + verdictColor(c.verdict) + ';opacity:0.85"></rect>';
    svg += '<text x="' + (x0 + w + 8) + '" y="' + (y + 15) + '" style="fill:var(--ink);' + mono + '">'
      + c.val.toFixed(4) + "  dsr " + c.dsr.toFixed(3) + "</text>";
  });
  svg += "</svg>";
  box.innerHTML = svg;
}

/* ── 挖掘 ─────────────────────────────────────────────────────────── */
function setStatus(text, on) {
  $("statusText").textContent = text;
  var dot = $("statusPill").querySelector(".dot");
  if (dot) { dot.className = on ? "dot on" : "dot"; }
}

function startMine() {
  var data = $("data").value;
  if (!data) { alert("请先选择行情数据文件"); return; }
  post("/api/mine", {
    data: data,
    budget: $("budget").value,
    seed: parseInt($("seed").value, 10) || 0,
    symbol: $("symbol").value.trim(),
    timeframe: $("timeframe").value.trim(),
    market: $("market").value
  }).then(function (r) {
    state.cmd = r.cmd || "";
    $("cmdBox").value = "python -m miaosuan.cli " + (r.cmd || "");
    state.logOffset = 0;
    state.candidates = [];
    $("log").textContent = "";
    startPolling();
  }).catch(function (e) { alert(e.message); });
}

function startPolling() {
  if (state.polling) { return; }
  state.polling = setInterval(pollJob, 900);
  pollJob();
}

function pollJob() {
  jfetch("/api/job?offset=" + state.logOffset).then(function (s) {
    state.logOffset = s.total;
    if (s.lines && s.lines.length) {
      var log = $("log");
      log.textContent += s.lines.join("\n") + "\n";
      if (state.autoscroll) { log.scrollTop = log.scrollHeight; }
    }
    if (s.running) {
      setStatus("挖掘中", true);
      $("btnMine").disabled = true;
    } else {
      setStatus("空闲", false);
      $("btnMine").disabled = false;
      if (state.polling && s.total > 0) {
        clearInterval(state.polling);
        state.polling = null;
        finishJob(s);
      }
    }
    // 进度区随日志实时更新
    var all = $("log").textContent.split("\n");
    var parsed = parseLog(all);
    state.candidates = parsed.candidates;
    renderChart(parsed.candidates);
    fillProgress(parsed.stats, s.running);
  }).catch(function () { /* 轮询失败静默，避免打断 */ });
}

function fillProgress(st, running) {
  $("pReason").textContent = st.reason;
  $("pGen").textContent = st.gen;
  $("pEval").textContent = st.evalN;
  $("pDiv").textContent = st.div;
  $("pHoldout").textContent = st.holdout;
  // 状态以**后端 running 标志**为准：轮询定时器空闲时也在跑，不能拿它当"挖掘中"。
  $("pState").textContent = running ? "挖掘中" : (st.reason === "—" ? "空闲" : "已完成");
}

function finishJob(s) {
  // 挖掘结束刷新文件列表：重算并重绘每个文件的时间范围（保留原选中）
  loadData();
  // 训练曲线：读 CLI 落盘的逐代历史 sidecar（机器可读，不污染 stdout 日志）。
  if (s.returncode === 0 && s.history_out) {
    jfetch("/api/history?name=" + encodeURIComponent(s.history_out)).then(function (h) {
      renderCurve(h.points || []);
    }).catch(function (e) {
      dbg("历史读取失败：" + e.message);
      renderCurve([]);   // 读取失败同样降级为空状态，不抛错到页面
    });
  } else {
    renderCurve([]);
  }
  if (s.returncode === 0 && s.spec_out) {
    jfetch("/api/spec/" + encodeURIComponent(s.spec_out)).then(function (spec) {
      state.spec = s.spec_out;
      renderSpec(spec);
      refreshLists();
    }).catch(function (e) { dbg("spec 加载失败：" + e.message); refreshLists(); });
  } else {
    refreshLists();
  }
}

function renderSpec(s) {
  var ev = s.evidence || {}, pv = s.provenance || {};
  $("pVal").textContent = ev.val_score === undefined || ev.val_score === null ? "—" : Number(ev.val_score).toFixed(4);
  $("pDsr").textContent = ev.deflated_sharpe === undefined || ev.deflated_sharpe === null ? "—" : Number(ev.deflated_sharpe).toFixed(4);
  $("pGate").innerHTML = verdictChip(ev.gate_verdict);
  $("formulaBox").textContent = s.formula_full || "（无公式）";

  $("lGate").innerHTML = verdictChip(ev.gate_verdict);
  var reasons = ev.gate_reasons || [];
  $("lReasons").textContent = reasons.length ? reasons.join(" / ") : "（无）";
  $("lGit").textContent = pv.git_sha || "—";
  $("lSeed").textContent = pv.seed === undefined || pv.seed === null ? "—" : pv.seed;
  $("lVocab").textContent = pv.vocab_version || "—";
  $("lMarket").textContent = pv.market || "—";
  $("lFp").textContent = pv.data_fingerprint || "—";
  refreshLists();
}

function verdictChip(v) {
  var cls = v === "DEPLOYABLE" ? "ok" : (v === "BLOCKED" ? "bad" : "warn");
  return '<span class="chip ' + cls + '">' + esc(v || "-") + "</span>";
}

/* ── 复制 ─────────────────────────────────────────────────────────── */
function copyFormula() {
  var text = $("formulaBox").textContent || "";
  if (!text || text === "（尚无结果）") { $("copyHint").textContent = "没有可复制的公式"; return; }
  var done = function () { $("copyHint").textContent = "已复制"; };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done, fallbackCopy);
  } else { fallbackCopy(); }
  function fallbackCopy() {
    var ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); done(); }
    catch (e) { $("copyHint").textContent = "复制失败，请手动选中"; }
    document.body.removeChild(ta);
  }
}

/* ── 列表 ─────────────────────────────────────────────────────────── */
function refreshLists() {
  jfetch("/api/specs").then(function (specs) {
    if (!specs.length) { $("specList").className = "empty"; $("specList").textContent = "（还没有 spec —— 先跑一次挖掘）"; return; }
    $("specList").className = "";
    $("specList").innerHTML = '<div class="tbl-wrap"><table><tr><th>文件</th><th>品种</th><th>周期</th>'
      + "<th>val</th><th>DSR</th><th>门禁</th><th>时间</th><th></th></tr>"
      + specs.map(function (s) {
        return '<tr><td class="mono tiny">' + esc(s.file) + "</td><td>" + esc(s.symbol || "-") + "</td>"
          + "<td>" + esc(s.timeframe || "-") + "</td>"
          + "<td>" + (s.val_score === null ? "-" : Number(s.val_score).toFixed(4)) + "</td>"
          + "<td>" + (s.deflated_sharpe === null ? "-" : Number(s.deflated_sharpe).toFixed(4)) + "</td>"
          + "<td>" + verdictChip(s.gate_verdict) + "</td>"
          + '<td class="tiny muted">' + esc(s.mtime || "") + "</td>"
          + '<td><button class="ghost" data-file="' + esc(s.file) + '">查看</button></td></tr>';
      }).join("") + "</table></div>";
  }).catch(function (e) {
    $("specList").className = "empty"; $("specList").textContent = "加载失败：" + e.message;
  });

  jfetch("/api/strategies").then(function (rows) {
    var sel = $("verifyFile");
    if (!rows.length) {
      $("stratList").className = "empty"; $("stratList").textContent = "（还没有导出策略）";
      sel.innerHTML = '<option value="">（暂无）</option>';
      return;
    }
    sel.innerHTML = rows.map(function (s) {
      return '<option value="' + esc(s.file) + '">' + esc(s.file) + "</option>";
    }).join("");
    $("stratList").className = "";
    $("stratList").innerHTML = '<div class="tbl-wrap"><table><tr><th>文件</th><th>magic</th>'
      + "<th>lint</th><th>时间</th><th></th></tr>"
      + rows.map(function (s) {
        var l = '<span class="chip ' + (s.lint_errors ? "bad" : "ok") + '">'
          + s.lint_errors + "E / " + s.lint_warnings + "W</span>";
        return '<tr><td class="mono tiny">' + esc(s.file) + '</td><td class="mono">' + esc(s.magic) + "</td>"
          + "<td>" + l + '</td><td class="tiny muted">' + esc(s.mtime) + "</td>"
          + '<td><a href="/api/file/' + encodeURIComponent(s.file) + '" download>下载</a></td></tr>';
      }).join("") + "</table></div>";
  }).catch(function (e) {
    $("stratList").className = "empty"; $("stratList").textContent = "加载失败：" + e.message;
  });
}

function loadSpecByName(file) {
  jfetch("/api/spec/" + encodeURIComponent(file)).then(function (spec) {
    state.spec = file;
    renderSpec(spec);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }).catch(function (e) { alert(e.message); });
}

/* ── 校验 ─────────────────────────────────────────────────────────── */
function runVerify() {
  var f = $("verifyFile").value;
  if (!f) { alert("还没有已导出的策略文件"); return; }
  var out = $("verifyOut");
  out.style.display = "block";
  out.textContent = "校验中…";
  post("/api/verify", { file: f }).then(function (r) {
    out.textContent = r.output || ("（无输出，code=" + r.returncode + "）");
    refreshLists();
  }).catch(function (e) { out.textContent = "校验失败：" + e.message; });
}

/* ── 回测页 ───────────────────────────────────────────────────────── */
function loadBacktestOptions() {
  jfetch("/api/specs").then(function (specs) {
    if (!specs.length) {
      $("btSpec").innerHTML = '<option value="">（还没有 spec —— 先跑一次挖掘）</option>';
      return;
    }
    $("btSpec").innerHTML = specs.map(function (s) {
      return '<option value="' + esc(s.file) + '">' + esc(s.file)
        + " · " + esc(s.symbol || "-") + " " + esc(s.timeframe || "-") + "</option>";
    }).join("");
  }).catch(function (e) {
    $("btSpec").innerHTML = '<option value="">（spec 列表失败：' + esc(e.message) + "）</option>";
  });
  jfetch("/api/data").then(function (rows) {
    if (!rows.length) {
      $("btData").innerHTML = '<option value="">（未找到行情文件）</option>';
      return;
    }
    $("btData").innerHTML = rows.map(function (d) {
      return '<option value="' + esc(d.path) + '">' + esc(d.name) + " · "
        + esc(dataRangeText(d)) + " · " + d.size_mb + "MB</option>";
    }).join("");
  }).catch(function (e) {
    $("btData").innerHTML = '<option value="">（行情列表失败：' + esc(e.message) + "）</option>";
  });
  jfetch("/api/meta").then(function (m) {
    var html = '<option value="" selected>自动（数据画像 → spec 画像）</option>';
    (m.profiles || []).forEach(function (p) {
      html += '<option value="' + esc(p) + '">' + esc(p) + "</option>";
    });
    $("btMarket").innerHTML = html;
  }).catch(function () { /* meta 失败不阻塞 */ });
}

function runBacktest() {
  var spec = $("btSpec").value;
  var data = $("btData").value;
  if (!spec) { alert("请先选择一个 StrategySpec（先在 01 挖掘页跑一次）"); return; }
  if (!data) { alert("请选择行情数据文件"); return; }
  var btn = $("btnBacktest");
  btn.disabled = true;
  $("btNote").textContent = "回测中…（子进程调 CLI，全样本逐 bar 计算）";
  post("/api/backtest", {
    spec: spec,
    data: data,
    market: $("btMarket").value,
    rolling_window: parseInt($("btWindow").value, 10) || 0
  }).then(function (r) {
    renderBacktest(r);
    $("btNote").textContent = "完成 · 结果已落盘 " + esc(r.file || "");
  }).catch(function (e) {
    $("btNote").textContent = "回测失败：" + e.message;
    alert("回测失败：" + e.message);
  }).then(function () {
    btn.disabled = false;
  });
}

/* 通用序列折线图（内联 SVG）。null 表示"该点无数据"（如滚动窗口未满），
   不做插值：遇到 null 就断线。 */
function renderSeries(boxId, values, opts) {
  var box = $(boxId);
  var vals = (values || []).slice();
  var has = vals.map(function (v) { return v !== null && v !== undefined && isFinite(v); });
  var n = vals.length;
  if (!n || !has.some(Boolean)) {
    box.innerHTML = '<div class="empty">' + esc(opts.emptyText || "暂无数据") + "</div>";
    return;
  }
  var W = 640, H = opts.height || 230, padL = 62, padR = 16, padT = 16, padB = 34;
  var innerW = W - padL - padR, innerH = H - padT - padB;
  var lo = Infinity, hi = -Infinity;
  for (var i = 0; i < n; i++) {
    if (!has[i]) { continue; }
    if (vals[i] < lo) { lo = vals[i]; }
    if (vals[i] > hi) { hi = vals[i]; }
  }
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-12) { hi = lo + 1; }
  if (opts.zeroLine) {
    if (lo > 0) { lo = 0; }
    if (hi < 0) { hi = 0; }
  }
  var pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  function xOf(i) { return padL + (n > 1 ? i / (n - 1) * innerW : innerW / 2); }
  function yOf(v) { return padT + (hi - v) / (hi - lo) * innerH; }

  var svg = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" height="' + H + '" role="img"'
    + ' aria-label="' + esc(opts.label || "序列") + '">';
  var mono = 'font-family:var(--mono);font-size:10px';
  for (var g = 0; g <= 4; g++) {
    var v = lo + (hi - lo) * g / 4, y = yOf(v);
    svg += '<line x1="' + padL + '" y1="' + y.toFixed(2) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(2)
      + '" style="stroke:var(--line);stroke-width:1"></line>';
    svg += '<text x="' + (padL - 6) + '" y="' + (y + 3).toFixed(2) + '" text-anchor="end"'
      + ' style="fill:var(--ink-2);' + mono + '">' + v.toFixed(opts.digits === undefined ? 3 : opts.digits) + "</text>";
  }
  if (opts.zeroLine && lo < 0 && hi > 0) {
    var yz = yOf(0);
    svg += '<line x1="' + padL + '" y1="' + yz.toFixed(2) + '" x2="' + (W - padR) + '" y2="' + yz.toFixed(2)
      + '" style="stroke:var(--ink-2);stroke-width:1;stroke-dasharray:4 3"></line>';
  }
  var d = "", pen = false;
  for (var k = 0; k < n; k++) {
    if (!has[k]) { pen = false; continue; }
    d += (pen ? " L" : "M") + xOf(k).toFixed(2) + " " + yOf(vals[k]).toFixed(2);
    pen = true;
  }
  svg += '<path d="' + d + '" fill="none" stroke="' + opts.color + '" stroke-width="2"'
    + ' stroke-linejoin="round" stroke-linecap="round"></path>';
  // X 轴：首 / 中 / 末 的下标
  [0, Math.floor((n - 1) / 2), n - 1].forEach(function (i) {
    svg += '<text x="' + xOf(i).toFixed(2) + '" y="' + (H - padB + 16) + '" text-anchor="middle"'
      + ' style="fill:var(--ink-2);' + mono + '">' + i + "</text>";
  });
  svg += "</svg>";
  box.innerHTML = svg;
}

function fmtPct(v) {
  return (v === null || v === undefined || !isFinite(v)) ? "—" : (v * 100).toFixed(2) + "%";
}
function fmtNum(v, digits) {
  return (v === null || v === undefined || !isFinite(v)) ? "—" : Number(v).toFixed(digits === undefined ? 4 : digits);
}

function renderBacktest(r) {
  var s = r.summary || {}, m = r.meta || {};
  $("kTotal").textContent = fmtPct(s.total_return);
  $("kAnn").textContent = fmtPct(s.annual_return);
  $("kSharpe").textContent = fmtNum(s.sharpe);
  $("kSortino").textContent = fmtNum(s.sortino);
  $("kMdd").textContent = fmtPct(s.max_drawdown);
  $("kTrades").textContent = String(s.n_trades === undefined ? "—" : s.n_trades);
  $("kWin").textContent = fmtPct(s.win_rate);
  $("kPf").textContent = (s.profit_factor === null || s.profit_factor === undefined)
    ? "无亏损段" : fmtNum(s.profit_factor);
  var _cBuy = Number(m.cost_rate).toFixed(6);
  var _cSell = (m.cost_rate_sell === undefined || m.cost_rate_sell === null)
    ? _cBuy : Number(m.cost_rate_sell).toFixed(6);
  $("btMeta").textContent = m.symbol + " " + m.timeframe + " · " + m.n_bars + " 根 · 单边成本 买 "
    + _cBuy + " / 卖 " + _cSell + " · 年化 bar " + m.periods_per_year
    + " · 滚动窗口 " + m.rolling_window;

  renderSeries("btEquity", (r.series || {}).equity, {
    color: "var(--gold)", label: "资金曲线", digits: 3,
    emptyText: "暂无资金曲线"
  });
  $("btEquityNote").innerHTML = "口径：权益 = <span class=\"mono\">1 + cumsum(逐 bar 净收益)</span>，"
    + "净收益已扣成本。回撤峰值（绝对，权益单位）<b>" + fmtNum(s.max_drawdown_abs, 4)
    + "</b>；概览卡里的" + "最大回撤是<b>相对</b>值 (peak−equity)/peak = " + fmtPct(s.max_drawdown)
    + " —— 两者单位不同，别混着比。";

  renderSeries("btRolling", (r.series || {}).rolling_sharpe, {
    color: "var(--info)", label: "滚动 Sharpe", digits: 3, zeroLine: true,
    emptyText: "暂无滚动 Sharpe（样本可能短于滚动窗口）"
  });
  $("btRollingNote").textContent = "口径：滑窗内均值/标准差 × √年化bar，窗口 " + m.rolling_window
    + " 根；窗口未满的前段留空（不补零、不插值）。";

  var trades = r.trades || [];
  if (!trades.length) {
    $("btTrades").className = "empty";
    $("btTrades").textContent = "本次回测没有产生任何交易（仓位全程未过中性带）。";
    return;
  }
  $("btTrades").className = "";
  var rows = trades.slice(0, 50).map(function (t) {
    return '<tr><td>' + t.start + "–" + t.end + "</td>"
      + '<td><span class="chip ' + (t.direction > 0 ? "ok" : "bad") + '">'
      + (t.direction > 0 ? "多" : "空") + "</span></td>"
      + "<td>" + t.bars + "</td>"
      + '<td style="color:' + (t.pnl >= 0 ? "var(--up)" : "var(--down)") + '">' + t.pnl.toFixed(6) + "</td></tr>";
  }).join("");
  $("btTrades").innerHTML = '<div class="tbl-wrap"><table><tr><th>区间(bar)</th><th>方向</th>'
    + "<th>持仓 bar</th><th>净收益(已扣成本)</th></tr>" + rows + "</table></div>"
    + '<div class="tiny muted" style="margin-top:6px">共 ' + s.n_trades + " 笔，此处显示前 50 笔"
    + (r.trades_truncated ? "（明细已截断）" : "") + "。</div>";
}

/* ── 04 调优页：参数寻优（TPE 采样 + 回测评估）─────────────────
   后端为长任务（Optuna 多次回测），采用「启动 → 轮询状态 → 取结果」三步，
   避免单请求挂死。结果落盘 artifacts/tune_*.json，前端拉取展示。 */
function loadTuneOptions() {
  jfetch("/api/specs").then(function (specs) {
    if (!specs.length) {
      $("tuneSpec").innerHTML = '<option value="">' + esc(t("noSpec")) + "</option>";
      return;
    }
    $("tuneSpec").innerHTML = specs.map(function (s) {
      return '<option value="' + esc(s.file) + '">' + esc(s.file)
        + " · " + esc(s.symbol || "-") + " " + esc(s.timeframe || "-") + "</option>";
    }).join("");
  }).catch(function (e) {
    $("tuneSpec").innerHTML = '<option value="">（spec 列表失败：' + esc(e.message) + "）</option>";
  });
  jfetch("/api/data").then(function (rows) {
    if (!rows.length) {
      $("tuneData").innerHTML = '<option value="">' + esc("未找到行情文件") + "</option>";
      return;
    }
    $("tuneData").innerHTML = rows.map(function (d) {
      return '<option value="' + esc(d.path) + '">' + esc(d.name) + " · "
        + esc(dataRangeText(d)) + " · " + d.size_mb + "MB</option>";
    }).join("");
  }).catch(function (e) {
    $("tuneData").innerHTML = '<option value="">（行情列表失败：' + esc(e.message) + "）</option>";
  });
  loadTuneHistory();
}

function startTune() {
  var spec = $("tuneSpec").value;
  var data = $("tuneData").value;
  if (!spec) { alert(t("noSpec")); return; }
  if (!data) { alert("请选择行情数据文件"); return; }
  var btn = $("btnStartTune");
  btn.disabled = true;
  $("tuneNote").textContent = t("tuneRunning");
  post("/api/tune/start", {
    spec: spec,
    data: data,
    param_spaces: [],   // 缺省：后端自动从 spec 推导（预留接口，暂无需手填）
    n_trials: parseInt($("tuneTrials").value, 10) || 50,
    metric: $("tuneMetric").value,
    seed: parseInt($("tuneSeed").value, 10) || 42
  }).then(function () {
    pollTune();
  }).catch(function (e) {
    $("tuneNote").textContent = "调优启动失败：" + e.message;
    btn.disabled = false;
  });
}

function pollTune() {
  var timer = setInterval(function () {
    jfetch("/api/tune/status").then(function (s) {
      if (s.running) {
        $("tuneNote").textContent = t("tuneRunning") + " …";
        return;
      }
      clearInterval(timer);
      $("btnStartTune").disabled = false;
      if (s.error) {
        $("tuneNote").textContent = "调优失败：" + s.error;
        return;
      }
      return loadTuneResult();
    }).catch(function (e) {
      clearInterval(timer);
      $("btnStartTune").disabled = false;
      $("tuneNote").textContent = "状态查询失败：" + e.message;
    });
  }, 2000);
}

function loadTuneResult() {
  jfetch("/api/tune/result").then(function (r) {
    $("tuneNote").textContent = t("tuneComplete");
    renderTuneResult(r);
    loadTuneHistory();
  }).catch(function (e) {
    $("tuneNote").textContent = "结果获取失败：" + e.message;
  });
}

function renderTuneResult(r) {
  var card = $("tuneResultCard");
  card.style.display = "";
  $("tBestScore").textContent = r.best ? Number(r.best.score).toFixed(4) : "—";
  $("tElapsed").textContent = (r.elapsed_sec || 0).toFixed(1) + "s";
  $("tTrials").textContent = String(r.n_trials || 0);
  // 参数表：最优参数逐行列出。值来自后端 JSON，统一先 esc() 再拼进单元格，
  // 避免把原始字符串直接塞进 innerHTML（参数名理论上可控，仍按注入面处理）。
  var bestParams = (r.best && r.best.params) || {};
  var keys = Object.keys(bestParams);
  var rows = keys.length ? keys.map(function (k) {
    return "<tr><td class=\"mono\">" + esc(k) + "</td>"
      + '<td class="mono">' + esc(String(Number(bestParams[k]).toFixed(6))) + "</td></tr>";
  }).join("") : '<tr><td colspan="2" class="muted tiny">（无参数字段）</td></tr>';
  var th1 = esc(t("paramTable"));
  var th2 = esc(t("paramValue"));
  $("tuneTable").innerHTML = '<tr><th>' + th1 + '</th><th>' + th2 + '</th></tr>' + rows;
}

function loadTuneHistory() {
  jfetch("/api/tune/list").then(function (rows) {
    var box = $("tuneHistory");
    if (!rows.length) {
      box.className = "empty";
      box.textContent = "（还没有调优记录）";
      return;
    }
    box.className = "";
    box.innerHTML = '<div class="tbl-wrap"><table><tr><th>文件</th><th>试验数</th>'
      + "<th>最优分</th><th>耗时</th><th>时间</th></tr>"
      + rows.map(function (r) {
        return '<tr><td class="mono tiny">' + esc(r.file) + "</td>"
          + "<td>" + esc(r.n_trials) + "</td>"
          + "<td class=\"mono\">" + Number(r.best_score || 0).toFixed(4) + "</td>"
          + "<td>" + (r.elapsed_sec || 0).toFixed(1) + "s</td>"
          + '<td class="tiny muted">' + esc(r.mtime || "") + "</td></tr>";
      }).join("") + "</table></div>";
  }).catch(function (e) {
    $("tuneHistory").className = "empty";
    $("tuneHistory").textContent = "加载失败：" + e.message;
  });
}

/* ── 03 实时：妙算 后端**只读**接入 ─────────────────────────────
   本页只对妙算后端发 GET（/api/realtime/*），妙算后端再只读转发到 妙算。
   后端不可达 / 超时 / 桥接离线三种情况都在顶部横幅**醒目提示**，并把数据区
   显示为「不可用」，绝不显示 0 或空表让用户误读成真实行情。 */
var RT = { timer: null, busy: false, everyMs: 2000 };

function rtGet(path, params) {
  var q = "", k;
  if (params) {
    var parts = [];
    for (k in params) {
      if (Object.prototype.hasOwnProperty.call(params, k)) {
        parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
      }
    }
    if (parts.length) { q = "?" + parts.join("&"); }
  }
  return fetch(path + q, { method: "GET" }).then(function (r) {
    return r.json().catch(function () { return null; }).then(function (body) {
      if (!r.ok) { throw new Error((body && body.detail) ? body.detail : ("HTTP " + r.status)); }
      return body;   // 信封：{ok, backend, reason, error, status, data}
    });
  });
}

function rtReasonText(reason) {
  if (reason === "ok") { return "正常"; }
  if (reason === "timeout") { return "请求超时"; }
  if (reason === "unreachable") { return "后端不可达"; }
  if (reason === "http_error") { return "后端返回错误"; }
  if (reason === "bad_json") { return "响应非 JSON"; }
  return "未知原因";
}

function rtBanner(kind, title, msg) {
  $("rtBanner").className = "banner " + kind;   // ok | warn | err
  $("rtBannerTitle").textContent = title;
  $("rtBannerMsg").textContent = msg;
}

function rtKv(label, value, cls) {
  return '<div><span class="tiny muted">' + esc(label) + '</span>'
    + '<div class="mono ' + (cls || "") + '">' + esc(value) + "</div></div>";
}

function rtUptime(sec) {
  var s = Math.max(0, Math.floor(Number(sec) || 0));
  if (s >= 3600) { return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m"; }
  if (s >= 60) { return Math.floor(s / 60) + "m " + (s % 60) + "s"; }
  return s + "s";
}

function rtTime(t) {
  if (t === undefined || t === null || t === "") { return "—"; }
  if (typeof t === "number") {
    return new Date(t > 1e12 ? t : t * 1000).toLocaleString();
  }
  return String(t).replace("T", " ").slice(0, 19);
}

function rtStatusText(s) {
  var base = s.status ? String(s.status) : "—";
  if (s.void_reason) { return base + "（" + s.void_reason + "）"; }
  if (s.exit_reason) { return base + "（" + s.exit_reason + "）"; }
  return base;
}

function renderRtEngine(env) {
  var box = $("rtEngine"), empty = $("rtEngineEmpty");
  var d = (env && env.ok && env.data && typeof env.data === "object") ? env.data : null;
  if (!d) {
    box.innerHTML = "";
    empty.style.display = "block";
    empty.textContent = "引擎状态不可用：" + (env ? rtReasonText(env.reason) + (env.error ? "（" + env.error + "）" : "") : "未获取");
    return null;
  }
  empty.style.display = "none";
  var bridge = d.bridge_connected === true;
  box.innerHTML =
      rtKv("运行状态", d.status === undefined ? "—" : d.status, String(d.status).toLowerCase() === "running" ? "up" : "down")
    + rtKv("运行时长", d.uptime_seconds === undefined ? "—" : rtUptime(d.uptime_seconds))
    + rtKv("桥接连接", d.bridge_connected === undefined ? "—" : (bridge ? "在线" : "离线"), bridge ? "up" : "down")
    + rtKv("启动时刻", d.started_at || "—");
  return { bridge: bridge };
}

function renderRtQuote(env) {
  var box = $("rtQuote"), empty = $("rtQuoteEmpty");
  var d = (env && env.ok && env.data && typeof env.data === "object") ? env.data : null;
  if (!d || d.bid === undefined) {
    box.innerHTML = "";
    empty.style.display = "block";
    empty.textContent = "报价不可用：" + (env ? rtReasonText(env.reason) + (env.error ? "（" + env.error + "）" : "") : "未获取");
    return;
  }
  empty.style.display = "none";
  function q(label, v) {
    return '<div class="q"><span>' + esc(label) + "</span><b>" + esc(v === undefined || v === null ? "—" : v) + "</b></div>";
  }
  box.innerHTML = q("BID", d.bid) + q("ASK", d.ask) + q("SPREAD", d.spread) + q("品种", d.symbol || "—");
}

function renderRtChart(env) {
  var data = (env && env.ok && Array.isArray(env.data)) ? env.data : [];
  var closes = [];
  for (var i = 0; i < data.length; i++) {
    var v = data[i] && data[i].close;
    closes.push(typeof v === "number" ? v : parseFloat(v));
  }
  var reason = env ? rtReasonText(env.reason) : "未获取";
  renderSeries("rtChart", closes, {
    label: "K 线收盘价",
    color: "var(--gold)",
    height: 200,
    digits: 2,
    emptyText: (env && env.ok)
      ? "后端已连接，但未返回 K 线。"
      : ("K 线不可用：" + reason + (env && env.error ? "（" + env.error + "）" : ""))
  });
  if (env && env.ok && closes.length) {
    $("rtChartCaption").textContent =
      "数据来源：GET /api/market/candles（真实 K 线 " + closes.length + " 根，非模拟）。";
  }
}

function renderRtSignals(env) {
  var box = $("rtSignals");
  if (!env || !env.ok) {
    box.className = "empty";
    box.textContent = "信号不可用：" + (env ? rtReasonText(env.reason) + (env.error ? "（" + env.error + "）" : "") : "未获取");
    return;
  }
  var rows = Array.isArray(env.data) ? env.data : [];
  if (!rows.length) {
    box.className = "empty";
    box.textContent = "后端已连接，但暂无信号。";
    return;
  }
  box.className = "";
  var out = '<div class="tbl-wrap"><table><tr><th>时间</th><th>策略</th><th>Magic</th><th>方向</th>'
    + "<th>多头分</th><th>空头分</th><th>状态</th><th>Ticket</th><th>入场价</th></tr>";
  rows.slice(0, 50).forEach(function (s) {
    function cell(v) { return v === undefined || v === null ? "—" : v; }
    out += "<tr>"
      + '<td class="mono">' + esc(rtTime(s.timestamp)) + "</td>"
      + "<td>" + esc(cell(s.strategy)) + "</td>"
      + '<td class="mono">' + esc(cell(s.magic)) + "</td>"
      + "<td>" + esc(cell(s.signal)) + "</td>"
      + '<td class="mono">' + esc(cell(s.score_long)) + "</td>"
      + '<td class="mono">' + esc(cell(s.score_short)) + "</td>"
      + "<td>" + esc(rtStatusText(s)) + "</td>"
      + '<td class="mono">' + esc(cell(s.ticket)) + "</td>"
      + '<td class="mono">' + esc(cell(s.price_entry)) + "</td>"
      + "</tr>";
  });
  out += "</table></div><div class='tiny muted' style='margin-top:6px'>共 " + rows.length
    + " 条，此处显示前 50 条。本页只读，不提供任何交易操作。</div>";
  box.innerHTML = out;
}

function rtRefresh() {
  if (RT.busy) { return Promise.resolve(); }
  RT.busy = true;
  return Promise.all([
    rtGet("/api/realtime/status"),
    rtGet("/api/realtime/price"),
    rtGet("/api/realtime/candles", { symbol: "XAUUSD", timeframe: "H1", count: 120 }),
    rtGet("/api/realtime/signals")
  ]).then(function (res) {
    RT.busy = false;
    var st = res[0] || {}, pr = res[1] || {}, cd = res[2] || {}, sg = res[3] || {};
    $("rtBackend").textContent = st.backend || pr.backend || "—";
    var meta = renderRtEngine(st);
    renderRtQuote(pr);
    renderRtChart(cd);
    renderRtSignals(sg);
    if (!st.ok) {
      rtBanner("err", "后端" + rtReasonText(st.reason),
        (st.error || "") + "  ·  " + (st.backend || "—") + "  ·  数据区已置为不可用，不显示任何行情数字以免误读");
    } else if (meta && !meta.bridge) {
      rtBanner("warn", "桥接离线", "引擎在运行，但 MT4 桥接未连接（bridge_connected=false）—— 报价/信号可能过期");
    } else {
      rtBanner("ok", "已连接（只读）", (st.backend || "—") + "  ·  仅 GET 白名单，无任何写操作");
    }
  }).catch(function (err) {
    RT.busy = false;
    rtBanner("err", "后端不可达", String(err && err.message ? err.message : err) + " —— 本页只读，不显示任何行情数字");
    $("rtEngine").innerHTML = "";
    $("rtEngineEmpty").style.display = "block";
    $("rtEngineEmpty").textContent = "后端不可达，无数据。";
    $("rtQuote").innerHTML = "";
    $("rtQuoteEmpty").style.display = "block";
    $("rtQuoteEmpty").textContent = "后端不可达，无数据。";
    $("rtChart").innerHTML = '<div class="empty">后端不可达，无数据。</div>';
    $("rtSignals").className = "empty";
    $("rtSignals").textContent = "后端不可达，无数据。";
  });
}

function rtSetPolling(on) {
  if (RT.timer) { clearInterval(RT.timer); RT.timer = null; }
  if (on) {
    rtRefresh();
    RT.timer = setInterval(rtRefresh, RT.everyMs);   // HTTP 轮询（2s）
  }
}

/* ── 数据获取页：妙算本地库 / 网络下载 · 增量优先 ─────────────────
   三步同调优：列来源 → 拉缓存 → 触发增量获取（POST 落盘后再刷缓存 + 行情下拉）。 */
function loadAcqSources(profile, isRetry) {
  var url = "/api/acquisition/sources" + (profile ? ("?profile=" + encodeURIComponent(profile)) : "");
  jfetch(url).then(function (rows) {
    var box = $("acqSources");
    if (!box) return;
    if (!rows.length) {
      box.className = "empty";
      box.textContent = "（该市场画像暂无可用数据来源）";
      return;
    }
    var html = '<select id="acqSource">';
    html += '<option value="" selected>请选择单一来源</option>';
    var pending = false;
    rows.forEach(function (s) {
      // available === null = 后端尚未探测（接口非阻塞立即返回）。显示为「检测中」
      // 而非「不可用」，且**不 disabled**——否则用户在探测完成前根本点不了。
      var unknown = (s.available === null || s.available === undefined);
      if (unknown) pending = true;
      var suffix = unknown ? "（检测中…）" : (s.available ? "" : "（不可用）");
      var dis = (s.available === false) ? " disabled" : "";
      var label = s.source_type + " —— " + s.description + suffix;
      html += '<option value="' + esc(s.source_type) + '"' + dis + ">" + esc(label) + "</option>";
    });
    html += "</select>";
    box.className = "";
    box.innerHTML = html;
    // 探测在后端后台线程进行：延迟 1.5s **只补刷一次**，用真实结果换掉「检测中」。
    // 刻意不做轮询（isRetry 守卫）；探测最多几秒，一次足够。
    if (window.__acqProbeTimer) { clearTimeout(window.__acqProbeTimer); window.__acqProbeTimer = 0; }
    if (pending && !isRetry) {
      window.__acqProbeTimer = setTimeout(function () { loadAcqSources(profile, true); }, 1500);
    }
  }).catch(function (e) {
    var box = $("acqSources");
    if (!box) return;
    box.className = "empty";
    box.textContent = "加载来源失败：" + e.message;
  });
}

/* 追加目标：非空 = 下次「获取数据」把增量接到该文件后面（写入前后端做连续性校验） */
var acqAppendTarget = "";

function setAppendTarget(path, name) {
  acqAppendTarget = path || "";
  var box = $("acqAppendBar");
  if (!box) return;
  if (!acqAppendTarget) {
    box.style.display = "none";
    box.textContent = "";
    return;
  }
  box.style.display = "";
  box.innerHTML = "追加目标：<b class=\"mono\">" + esc(name || String(acqAppendTarget).split(/[\\/]/).pop())
    + "</b> — 新增量将接在该文件末尾（写入前自动校验品种/周期/时间连续性）"
    + ' <button class="ghost" id="btnClearAppend">取消追加</button>';
  $("btnClearAppend").onclick = function () { setAppendTarget("", ""); };
}

function loadAcqCached() {
  jfetch("/api/acquisition/cached").then(function (rows) {
    var box = $("acqCached");
    if (!rows.length) {
      box.className = "empty";
      box.textContent = t("noCache");
      return;
    }
    box.className = "";
    // 「时间」改名为「更新时间」：它渲染的是文件 mtime，与数据起止范围同名会误读。
    // 「操作」列：追加按钮把该文件设为增量追加目标（选中行高亮）。
    box.innerHTML = '<div class="tbl-wrap"><table><tr><th>文件</th><th>大小</th>'
      + "<th>数据范围</th><th>更新时间</th><th>操作</th></tr>"
      + rows.map(function (r) {
        var active = acqAppendTarget && acqAppendTarget === r.path;
        return '<tr' + (active ? ' class="append-active"' : "") + '><td class="mono tiny">' + esc(r.name) + "</td>"
          + "<td>" + (r.size_mb != null ? r.size_mb + "MB" : "—") + "</td>"
          + '<td class="tiny">' + esc(dataRangeText(r)) + "</td>"
          + '<td class="tiny muted">' + esc(r.mtime || r.updated_at || "") + "</td>"
          + '<td><button class="ghost tiny append-btn" data-path="' + esc(r.path)
          + '" data-name="' + esc(r.name) + '">' + (active ? "已选中" : "追加") + "</button></td></tr>";
      }).join("") + "</table></div>";
    // 事件委托：表格 innerHTML 每次刷新都会重写，监听器挂容器上（同 specList）。
    box.onclick = function (ev) {
      var btn = (ev.target && ev.target.closest) ? ev.target.closest("button.append-btn") : null;
      if (!btn) return;
      var p = btn.getAttribute("data-path");
      // 再点同一行 = 取消选中
      if (acqAppendTarget === p) { setAppendTarget("", ""); } else { setAppendTarget(p, btn.getAttribute("data-name")); }
      loadAcqCached();
    };
  }).catch(function (e) {
    var box = $("acqCached");
    box.className = "empty";
    box.textContent = "加载缓存失败：" + e.message;
  });
}

function startAcqFetch() {
  var profile = $("acqProfile").value;
  var symbol = $("acqSymbol").value.trim();
  var timeframe = $("acqTimeframe").value.trim();
  var source = ($("acqSource") ? $("acqSource").value : "") || "";
  var note = $("acqNote").value.trim();
  var since = parseInt($("acqSince").value, 10) || 0;

  if (!profile) { showModal("无法获取", "请先选择「市场画像」。", "err"); return; }
  if (!symbol) { showModal("无法获取", "请选择「品种」（选项来自市场画像）。", "err"); return; }
  if (!timeframe) { showModal("无法获取", "请选择「周期」（选项来自市场画像）。", "err"); return; }
  if (!source) { showModal("无法获取", "请选择「数据来源」（选定市场画像后，仅能选一个）。", "err"); return; }

  // 追加模式：弹确认，讲清楚「接到哪个文件、校验什么」
  if (acqAppendTarget) {
    var tgtName = String(acqAppendTarget).split(/[\\/]/).pop();
    var okAppend = confirm(
      "将把 " + symbol + " " + timeframe + " 的增量追加到：\n\n"
      + tgtName + "\n\n"
      + "写入前会自动校验：\n"
      + "· 文件名品种/周期与请求一致\n"
      + "· 数据实际周期与请求周期相符\n"
      + "· 时间戳单位一致、新数据延伸旧时间轴\n"
      + "· 新旧缺口不超过 30 天（超 7 天给出警告）\n\n"
      + "任何一项不通过都不会写入。继续？"
    );
    if (!okAppend) return;
  }

  var btn = $("btnAcqFetch");
  btn.disabled = true;
  $("acqStatus").textContent = acqAppendTarget
    ? "追加中…（增量接至目标文件末尾，写入前校验连续性）"
    : "获取中…（增量优先：本地无则全量，有则增量）";
  post("/api/acquisition/fetch", {
    symbol: symbol,
    timeframe: timeframe,
    since: since,
    source: source,
    note: note,
    market_profile: profile,
    append_to: acqAppendTarget
  }).then(function (r) {
    // 三色语义：updated/up_to_date=绿、stale_fallback=黄（附各源失败原因）
    var st = r.status || "updated";
    var lines = [];
    if (st === "stale_fallback") {
      lines.push("<b style=\"color:var(--warn)\">⚠ 所有数据源均失败，已回退本地旧缓存（数据可能过期）</b>");
      lines.push("旧文件：" + esc(r.path || "—"));
      if (r.errors && r.errors.length) {
        lines.push("失败原因：<br>· " + r.errors.map(esc).join("<br>· "));
      }
      showModal("获取失败 · 使用旧缓存", lines.join("<br>"), "warn");
      $("acqStatus").textContent = "⚠ 回退旧缓存：" + (r.path || "");
    } else if (st === "up_to_date") {
      lines.push("源已正常响应，本地缓存已是最新（无新 bar）。");
      lines.push("文件：" + esc(r.path || "—"));
      if (r.source_used) lines.push("来源：" + esc(r.source_used));
      if (r.warnings && r.warnings.length) {
        lines.push("部分源异常（不影响结果）：<br>· " + r.warnings.map(esc).join("<br>· "));
      }
      showModal("数据已是最新", lines.join("<br>"), "ok");
      $("acqStatus").textContent = "已最新：" + (r.path || "");
    } else {
      lines.push("已落盘：" + esc(r.path || "（未知路径）"));
      if (r.source_used) lines.push("来源：" + esc(r.source_used));
      if (r.note) lines.push("备注：" + esc(r.note));
      if (r.warnings && r.warnings.length) {
        lines.push("部分源异常（不影响结果）：<br>· " + r.warnings.map(esc).join("<br>· "));
      }
      showModal("数据获取成功", lines.join("<br>"), "ok");
      $("acqStatus").textContent = "完成：" + (r.path || "");
    }
    loadAcqCached();
    loadData(r.path || ""); // 刷新 01 的行情下拉，并优先选中刚落盘的文件
  }).catch(function (e) {
    showModal("数据获取失败", esc(e.message), "err");
    $("acqStatus").textContent = "获取失败：" + e.message;
  }).then(function () { btn.disabled = false; });
}

/* ── 数据获取页：市场画像联动 / Tab 切换 / 弹窗 ───────────────────── */
function onAcqProfileChange() {
  var p = $("acqProfile").value;
  var det = window.__profileDetails[p] || { symbols: [], timeframes: [] };
  $("acqSymbol").innerHTML = (det.symbols && det.symbols.length)
    ? det.symbols.map(function (s) { return '<option value="' + esc(s) + '">' + esc(s) + "</option>"; }).join("")
    : '<option value="">（该画像暂无预设品种）</option>';
  $("acqTimeframe").innerHTML = (det.timeframes && det.timeframes.length)
    ? det.timeframes.map(function (tf) { return '<option value="' + esc(tf) + '">' + esc(tf) + "</option>"; }).join("")
    : '<option value="">（该画像暂无预设周期）</option>';
  if (p) {
    loadAcqSources(p);
  } else {
    var box = $("acqSources");
    if (box) box.innerHTML = '<div class="empty">请先选择市场画像以解锁数据来源。</div>';
  }
}

function switchDataTab(tab) {
  var tabs = document.querySelectorAll("#dataCard .tab");
  for (var i = 0; i < tabs.length; i++) {
    tabs[i].classList.toggle("active", tabs[i].getAttribute("data-tab") === tab);
  }
  $("tab-sel").classList.toggle("active", tab === "sel");
  $("tab-acq").classList.toggle("active", tab === "acq");
}

function showModal(title, bodyHtml, kind) {
  $("modalTitle").textContent = title;
  $("modalTitle").className = "modal-title " + (kind || "");
  $("modalBody").innerHTML = bodyHtml;
  $("modal").classList.add("show");
}

function hideModal() {
  $("modal").classList.remove("show");
}

/* ── 05 Spec 管理 ─────────────────────────────────────────────────── */
var spmData = { specs: [], strategies: [], backtests: [], tunes: [], summary: {} };
var spmSelectedFile = null;

function loadSpecMgmt() {
  jfetch("/api/specs/detail").then(function (data) {
    spmData = data;
    renderSpmSummary();
    renderSpmList();
  }).catch(function (e) {
    $("spmList").className = "empty";
    $("spmList").textContent = "加载失败：" + e.message;
  });
}

function renderSpmSummary() {
  var s = spmData.summary || {};
  $("spmTotalSpecs").textContent = s.total_specs || 0;
  $("spmTotalStrategies").textContent = s.total_strategies || 0;
  $("spmTotalBacktests").textContent = s.total_backtests || 0;
  $("spmTotalTunes").textContent = s.total_tunes || 0;
  $("spmDeployable").textContent = s.deployable_count || 0;
  $("spmBlocked").textContent = s.blocked_count || 0;
  $("spmResearch").textContent = s.research_only_count || 0;
  $("spmExported").textContent = s.exported_count || 0;
}

function renderSpmList() {
  var search = ($("spmSearch").value || "").toLowerCase().trim();
  var filter = $("spmFilter").value;
  var specs = spmData.specs.filter(function (s) {
    // 搜索过滤
    if (search) {
      var haystack = ((s.name || "") + " " + (s.spec_id || "") + " " + (s.file || "") + " " + (s.symbol || "")).toLowerCase();
      if (haystack.indexOf(search) === -1) return false;
    }
    // 门禁/导出过滤
    if (filter === "DEPLOYABLE" && s.gate_verdict !== "DEPLOYABLE") return false;
    if (filter === "BLOCKED" && s.gate_verdict !== "BLOCKED") return false;
    if (filter === "RESEARCH_ONLY" && s.gate_verdict !== "RESEARCH_ONLY") return false;
    if (filter === "exported" && !(s.exports && s.exports.length)) return false;
    if (filter === "not_exported" && (s.exports && s.exports.length)) return false;
    return true;
  });

  if (!specs.length) {
    $("spmList").className = "empty";
    $("spmList").textContent = "（无匹配的 Spec）";
    return;
  }
  $("spmList").className = "";
  var html = '<div class="tbl-wrap"><table>'
    + '<tr><th>文件</th><th>名称</th><th>spec_id</th><th>品种</th><th>周期</th>'
    + '<th>val</th><th>DSR</th><th>门禁</th><th>导出</th><th>回测</th><th>寻优</th>'
    + '<th>时间</th><th></th></tr>';
  specs.forEach(function (s) {
    var nExp = (s.exports && s.exports.length) || 0;
    var nBt = (s.backtests && s.backtests.length) || 0;
    var nTu = (s.tunes && s.tunes.length) || 0;
    var isSelected = spmSelectedFile && s.file === spmSelectedFile;
    var rowClass = isSelected ? ' class="spm-selected"' : '';
    html += '<tr' + rowClass + '>'
      + '<td class="mono tiny">' + esc(s.file) + '</td>'
      + '<td>' + esc(s.name || "-") + '</td>'
      + '<td class="mono tiny">' + esc(s.spec_id ? s.spec_id.substring(0, 12) : "-") + '</td>'
      + '<td>' + esc(s.symbol || "-") + '</td>'
      + '<td>' + esc(s.timeframe || "-") + '</td>'
      + '<td>' + (s.val_score === null || s.val_score === undefined ? "-" : Number(s.val_score).toFixed(4)) + '</td>'
      + '<td>' + (s.deflated_sharpe === null || s.deflated_sharpe === undefined ? "-" : Number(s.deflated_sharpe).toFixed(4)) + '</td>'
      + '<td>' + verdictChip(s.gate_verdict) + '</td>'
      + '<td>' + (nExp ? '<span class="chip ok">' + nExp + '</span>' : '<span class="muted">0</span>') + '</td>'
      + '<td>' + (nBt ? '<span class="chip">' + nBt + '</span>' : '<span class="muted">0</span>') + '</td>'
      + '<td>' + (nTu ? '<span class="chip">' + nTu + '</span>' : '<span class="muted">0</span>') + '</td>'
      + '<td class="tiny muted">' + esc(s.mtime || "") + '</td>'
      + '<td><button class="ghost" data-spm-file="' + esc(s.file) + '">详情</button></td>'
      + '</tr>';
  });
  html += '</table></div>';
  $("spmList").innerHTML = html;
}

function showSpecDetail(file) {
  spmSelectedFile = file;
  $("spmDetailContainer").style.display = "";
  var spec = null;
  for (var i = 0; i < spmData.specs.length; i++) {
    if (spmData.specs[i].file === file) { spec = spmData.specs[i]; break; }
  }
  if (!spec) return;

  $("spmDetailTitle").innerHTML = "— <strong>" + esc(spec.name || spec.file) + "</strong>";

  // Payload
  $("spmSpecId").textContent = spec.spec_id || "—";
  $("spmNTokens").textContent = spec.n_tokens || "—";
  $("spmFormula").textContent = spec.formula || "—";
  $("spmVocab").textContent = spec.vocab_version || "—";
  $("spmFile").textContent = spec.file || "—";

  // Semantics
  var sem = spec.semantics || {};
  $("spmPosFn").textContent = sem.position_fn || "—";
  $("spmNeutralBand").textContent = sem.neutral_band !== undefined ? sem.neutral_band : "—";
  $("spmLongShort").textContent = sem.long_short ? "是（多空双向）" : "否（仅多头）";
  $("spmWarmup").textContent = sem.warmup_bars !== undefined ? sem.warmup_bars : "—";
  $("spmTimeframe").textContent = sem.timeframe || "—";
  $("spmRollWindow").textContent = sem.roll_window !== undefined ? sem.roll_window : "—";

  // Evidence
  var ev = spec.evidence || {};
  $("spmGate").innerHTML = verdictChip(ev.gate_verdict);
  var reasons = ev.gate_reasons || [];
  $("spmGateReasons").textContent = reasons.length ? reasons.join(" / ") : "（无）";
  $("spmValScore").textContent = ev.val_score !== undefined && ev.val_score !== null ? Number(ev.val_score).toFixed(4) : "—";
  $("spmDsr").textContent = ev.deflated_sharpe !== undefined && ev.deflated_sharpe !== null ? Number(ev.deflated_sharpe).toFixed(4) : "—";
  $("spmNTrials").textContent = ev.n_trials || "—";
  $("spmWfFolds").textContent = ev.wf_folds || "—";
  $("spmHoldout").textContent = ev.holdout_sharpe !== undefined && ev.holdout_sharpe !== null ? Number(ev.holdout_sharpe).toFixed(4) : "未消费";
  // 成本敏感度
  var cs = ev.cost_sensitivity || {};
  var csText = Object.keys(cs).map(function (k) {
    return k + "x → Sharpe " + Number(cs[k]).toFixed(2);
  }).join("   |   ");
  $("spmCostSensitivity").textContent = csText || "—";

  // Provenance
  var pv = spec.provenance || {};
  $("spmCreatedAt").textContent = pv.created_at || "—";
  $("spmGitSha").textContent = pv.git_sha || "—";
  $("spmDataFp").textContent = pv.data_fingerprint || "—";
  $("spmSeed").textContent = pv.seed !== undefined && pv.seed !== null ? pv.seed : "—";
  $("spmMarket").textContent = pv.market || "—";
  $("spmBudget").textContent = pv.budget || "—";

  // 谱系
  renderSpmLineage(spec);

  // 导出记录
  renderSpmExports(spec.exports || []);

  // 回测记录
  renderSpmBacktests(spec.backtests || []);

  // 寻优记录
  renderSpmTunes(spec.tunes || []);

  // 刷新列表以高亮选中行
  renderSpmList();

  // 滚动到详情区
  $("spmDetailContainer").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSpmLineage(spec) {
  var lineage = spec.lineage || [];
  var derivation = spec.derivation || "mine";
  var parent = spec.parent_spec_id;

  if (!parent && (!lineage || !lineage.length)) {
    // 根节点：查找是否有子 spec（其他 spec 的 parent 指向当前）
    var children = [];
    if (spmData.specs) {
      spmData.specs.forEach(function (s) {
        if (s.parent_spec_id && s.parent_spec_id === spec.spec_id) {
          children.push(s);
        }
      });
    }
    if (!children.length) {
      $("spmLineage").className = "empty";
      $("spmLineage").textContent = "当前 Spec 为根节点（mine 产出，无父 Spec，无派生子 Spec）。";
      return;
    }
    // 有子 spec：渲染树
    $("spmLineage").className = "";
    var html = '<div class="lineage-tree">';
    html += '<div class="lineage-node current">'
      + '<span class="mono">' + esc(spec.name || spec.file) + '</span>'
      + '<span class="lineage-detail">spec_id: ' + esc((spec.spec_id || "").substring(0, 12)) + '</span>'
      + verdictChip(spec.gate_verdict)
      + ' <span class="lineage-detail">(' + esc(derivation) + ')</span>'
      + '</div>';
    html += renderSpmChildren(spec, 1);
    html += '</div>';
    $("spmLineage").innerHTML = html;
    return;
  }

  // 有父 spec：渲染谱系链
  $("spmLineage").className = "";
  var html = '<div class="lineage-tree">';
  if (lineage && lineage.length) {
    lineage.forEach(function (parentId) {
      var parentSpec = findSpecById(parentId);
      html += '<div class="lineage-node' + (parentSpec ? ' clickable' : '') + '"'
        + (parentSpec ? ' data-spm-file="' + esc(parentSpec.file) + '"' : '') + '>'
        + '<span class="mono">' + esc(parentSpec ? parentSpec.name : parentId.substring(0, 12)) + '</span>'
        + (parentSpec ? verdictChip(parentSpec.gate_verdict) : '')
        + '</div>';
      html += '<div class="lineage-arrow">↓</div>';
    });
  }
  html += '<div class="lineage-node current">'
    + '<span class="mono">' + esc(spec.name || spec.file) + '</span>'
    + '<span class="lineage-detail">spec_id: ' + esc((spec.spec_id || "").substring(0, 12)) + '</span>'
    + verdictChip(spec.gate_verdict)
    + ' <span class="lineage-detail">(' + esc(derivation) + ')</span>'
    + '</div>';
  // 查找子 spec
  html += renderSpmChildren(spec, 1);
  html += '</div>';
  $("spmLineage").innerHTML = html;
}

function renderSpmChildren(parentSpec, depth) {
  if (depth > 10) return ""; // 防止无限递归
  var children = [];
  if (spmData.specs) {
    spmData.specs.forEach(function (s) {
      if (s.parent_spec_id && s.parent_spec_id === parentSpec.spec_id) {
        children.push(s);
      }
    });
  }
  if (!children.length) return "";
  var html = '<div class="lineage-children">';
  children.forEach(function (child) {
    html += '<div class="lineage-node clickable" data-spm-file="' + esc(child.file) + '">'
      + '<span class="mono">' + esc(child.name || child.file) + '</span>'
      + '<span class="lineage-detail">spec_id: ' + esc((child.spec_id || "").substring(0, 12)) + '</span>'
      + verdictChip(child.gate_verdict)
      + ' <span class="lineage-detail">(' + esc(child.derivation || "mine") + ')</span>'
      + '</div>';
    html += renderSpmChildren(child, depth + 1);
  });
  html += '</div>';
  return html;
}

function findSpecById(specId) {
  if (!specId || !spmData.specs) return null;
  for (var i = 0; i < spmData.specs.length; i++) {
    if (spmData.specs[i].spec_id && spmData.specs[i].spec_id.substring(0, 12) === specId.substring(0, 12)) {
      return spmData.specs[i];
    }
  }
  return null;
}

function renderSpmExports(exports) {
  if (!exports || !exports.length) {
    $("spmExports").className = "empty";
    $("spmExports").textContent = "无导出记录。";
    return;
  }
  $("spmExports").className = "";
  var html = '<div class="tbl-wrap"><table>'
    + '<tr><th>文件</th><th>magic</th><th>策略名</th><th>可部署</th><th>lint</th><th>大小</th><th>时间</th><th></th></tr>';
  exports.forEach(function (e) {
    var l = '<span class="chip ' + (e.lint_errors ? "bad" : "ok") + '">'
      + e.lint_errors + "E / " + e.lint_warnings + "W</span>";
    var dep = e.deployable ? '<span class="chip ok">是</span>' : '<span class="chip bad">否</span>';
    html += '<tr>'
      + '<td class="mono tiny">' + esc(e.file) + '</td>'
      + '<td class="mono">' + esc(e.magic) + '</td>'
      + '<td>' + esc(e.strategy_name || "-") + '</td>'
      + '<td>' + dep + '</td>'
      + '<td>' + l + '</td>'
      + '<td>' + esc(e.size_kb + "KB") + '</td>'
      + '<td class="tiny muted">' + esc(e.mtime) + '</td>'
      + '<td><a href="/api/file/' + encodeURIComponent(e.file) + '" download>下载</a></td>'
      + '</tr>';
  });
  html += '</table></div>';
  $("spmExports").innerHTML = html;
}

function renderSpmBacktests(backtests) {
  if (!backtests || !backtests.length) {
    $("spmBacktests").className = "empty";
    $("spmBacktests").textContent = "无回测记录。";
    return;
  }
  $("spmBacktests").className = "";
  var html = '<div class="tbl-wrap"><table>'
    + '<tr><th>文件</th><th>品种</th><th>周期</th><th>bar数</th><th>Sharpe</th>'
    + '<th>累计收益</th><th>最大回撤</th><th>Sortino</th><th>交易数</th><th>胜率</th><th>时间</th></tr>';
  backtests.forEach(function (bt) {
    html += '<tr>'
      + '<td class="mono tiny">' + esc(bt.file) + '</td>'
      + '<td>' + esc(bt.symbol || "-") + '</td>'
      + '<td>' + esc(bt.timeframe || "-") + '</td>'
      + '<td>' + esc(bt.n_bars || "-") + '</td>'
      + '<td>' + Number(bt.sharpe || 0).toFixed(4) + '</td>'
      + '<td>' + (Number(bt.total_return || 0) * 100).toFixed(2) + '%</td>'
      + '<td class="down">' + (Number(bt.max_drawdown || 0) * 100).toFixed(2) + '%</td>'
      + '<td>' + Number(bt.sortino || 0).toFixed(4) + '</td>'
      + '<td>' + esc(bt.n_trades || "-") + '</td>'
      + '<td>' + (Number(bt.win_rate || 0) * 100).toFixed(1) + '%</td>'
      + '<td class="tiny muted">' + esc(bt.mtime) + '</td>'
      + '</tr>';
  });
  html += '</table></div>';
  $("spmBacktests").innerHTML = html;
}

function renderSpmTunes(tunes) {
  if (!tunes || !tunes.length) {
    $("spmTunes").className = "empty";
    $("spmTunes").textContent = "无寻优记录。";
    return;
  }
  $("spmTunes").className = "";
  var html = '<div class="tbl-wrap"><table>'
    + '<tr><th>文件</th><th>result_id</th><th>试验数</th><th>最优评分</th><th>耗时</th>'
    + '<th>目标</th><th>最优参数</th><th>时间</th></tr>';
  tunes.forEach(function (tu) {
    var params = tu.best_params || {};
    var paramStr = Object.keys(params).map(function (k) {
      return k + "=" + (typeof params[k] === "number" ? Number(params[k]).toFixed(4) : params[k]);
    }).join(", ");
    html += '<tr>'
      + '<td class="mono tiny">' + esc(tu.file) + '</td>'
      + '<td class="mono tiny">' + esc(tu.result_id || "-") + '</td>'
      + '<td>' + esc(tu.n_trials || "-") + '</td>'
      + '<td>' + Number(tu.best_score || 0).toFixed(4) + '</td>'
      + '<td>' + esc((tu.elapsed_sec || 0) + "s") + '</td>'
      + '<td>' + esc(tu.metric || "-") + '</td>'
      + '<td class="mono tiny">' + esc(paramStr || "-") + '</td>'
      + '<td class="tiny muted">' + esc(tu.mtime) + '</td>'
      + '</tr>';
  });
  html += '</table></div>';
  $("spmTunes").innerHTML = html;
}

/* ── 绑定 ─────────────────────────────────────────────────────────── */
function bind() {
  $("btnTheme").onclick = function () {
    var cur = document.documentElement.getAttribute("data-theme");
    applyTheme(cur === "dark" ? "light" : "dark");
  };
  var btns = document.querySelectorAll("#stepper button");
  for (var i = 0; i < btns.length; i++) {
    btns[i].onclick = (function (btn) {
      return function () { setView(btn.getAttribute("data-view")); };
    })(btns[i]);
  }
  $("btnInspect").onclick = inspect;
  $("data").onchange = function () { inspect(); updateMineDataHint(); autoFillTimeframe(); };
  $("btnMine").onclick = startMine;
  $("btnCopy").onclick = copyFormula;
  $("btnVerify").onclick = runVerify;
  $("btnBacktest").onclick = runBacktest;
  $("btnStartTune").onclick = startTune;
  $("btnAcqFetch").onclick = startAcqFetch;
  $("acqProfile").onchange = onAcqProfileChange;
  $("btnRefreshCache").onclick = loadAcqCached;
  // 回测/调优页：数据文件切换时自动回填周期
  $("btData").onchange = function () { autoFillTimeframeForBt(); };
  $("tuneData").onchange = function () { autoFillTimeframeForTune(); };
  // 事件委托：specList 每次刷新都会重写 innerHTML，故监听器挂在容器上。
  // 用 data-file + getAttribute 取值（不再把文件名拼进 onclick 的 JS 字符串上下文，
  // 否则文件名里的引号能逃出 HTML 转义 → 注入）。
  $("specList").addEventListener("click", function (ev) {
    var btn = (ev.target && ev.target.closest) ? ev.target.closest("button[data-file]") : null;
    if (btn) { loadSpecByName(btn.getAttribute("data-file")); }
  });
  $("rtPoll").onchange = function () {
    if (state.view === "live") { rtSetPolling($("rtPoll").checked); }
  };
  $("btnScroll").onclick = function () {
    state.autoscroll = !state.autoscroll;
    $("btnScroll").textContent = state.autoscroll ? "暂停滚动" : "恢复滚动";
  };
  $("btnDebug").onclick = function () {
    state.debug = !state.debug;
    $("dbgOut").style.display = state.debug ? "block" : "none";
    $("btnDebug").textContent = state.debug ? "隐藏调试信息" : "显示调试信息";
    if (state.debug) { $("dbgOut").textContent = state.trace.join("\n") || "（暂无记录）"; }
  };
  // 05 Spec 管理
  $("btnSpmRefresh").onclick = loadSpecMgmt;
  $("spmSearch").oninput = renderSpmList;
  $("spmFilter").onchange = renderSpmList;
  $("spmList").addEventListener("click", function (ev) {
    var btn = (ev.target && ev.target.closest) ? ev.target.closest("button[data-spm-file]") : null;
    if (btn) { showSpecDetail(btn.getAttribute("data-spm-file")); }
  });
  $("spmLineage").addEventListener("click", function (ev) {
    var node = (ev.target && ev.target.closest) ? ev.target.closest(".lineage-node.clickable") : null;
    if (node) {
      var file = node.getAttribute("data-spm-file");
      if (file) { showSpecDetail(file); }
    }
  });
  window.onerror = function (msg, src, ln) {
    state.errors.push("JS: " + msg + " @" + src + ":" + ln);
    dbg("JS ERROR " + msg + " @" + src + ":" + ln);
  };
}

/* ── 06 设置页：加载/保存配置 ───────────────────────────────────────── */
function loadConfig() {
  var statusEl = $("cfgStatus");
  if (statusEl) { statusEl.textContent = "加载中…"; statusEl.className = "tiny muted"; }
  jfetch("/api/config").then(function (cfg) {
    // WebUI
    setVal("cfgWebUIHost", cfg.webui?.host || "127.0.0.1");
    setVal("cfgWebUIPort", cfg.webui?.port || 8686);
    setVal("cfgWebUIReload", String(cfg.webui?.reload || false));
    // Shenji
    setVal("cfgShenjiHost", cfg.shenji?.host || "127.0.0.1");
    setVal("cfgShenjiPort", cfg.shenji?.port || 1783);
    setVal("cfgShenjiTimeout", cfg.shenji?.timeout || 3.0);
    // MT4
    setVal("cfgMT4Host", cfg.mt4?.host || "127.0.0.1");
    setVal("cfgMT4Port", cfg.mt4?.port || 23232);
    setVal("cfgMT4TimeBase", cfg.mt4?.time_base || "utc");
    setVal("cfgMT4Timeout", cfg.mt4?.timeout || 5.0);
    setVal("cfgMT4PollWait", cfg.mt4?.poll_wait || 1000);
    setVal("cfgMT4DrainWait", cfg.mt4?.drain_wait || 100);
    // Paths
    setVal("cfgPathsDataCache", cfg.paths?.data_cache || "data/cache");
    setVal("cfgPathsArtifacts", cfg.paths?.artifacts || "artifacts");
    setVal("cfgPathsTmp", cfg.paths?.tmp || "tmp");
    setVal("cfgPathsKline", cfg.paths?.kline || "");
    setVal("cfgPathsShenjiDB", cfg.paths?.shenji_db || "");
    // Data
    setVal("cfgDataTimeout", cfg.data?.timeout || 30);
    setVal("cfgDataSource", cfg.data?.source || "");
    setVal("cfgDataAPIUrl", cfg.data?.api_url || "");
    setVal("cfgDataCacheDir", cfg.data?.cache_dir || "");
    setVal("cfgDataDukascopyUser", cfg.data?.dukascopy_user || "");
    setVal("cfgDataDukascopyPass", cfg.data?.dukascopy_password || "");
    // Logging
    setVal("cfgLoggingLevel", cfg.logging?.level || "INFO");
    setVal("cfgLoggingJSON", String(cfg.logging?.json_lines || true));
    // Features
    setVal("cfgFeatureWebUI", String(cfg.features?.enable_webui || true));
    setVal("cfgFeatureMT4", String(cfg.features?.enable_mt4 || true));
    setVal("cfgFeatureShenji", String(cfg.features?.enable_shenji || true));
    setVal("cfgFeatureSpec", String(cfg.features?.enable_spec_management || true));
    setVal("cfgFeatureBacktest", String(cfg.features?.enable_backtest || true));
    setVal("cfgFeatureTune", String(cfg.features?.enable_tune || true));
    // Config file path
    var manager = cfg._manager || {};
    if (statusEl) {
      statusEl.textContent = "已加载（来自 settings.yaml + 环境变量）";
      statusEl.className = "tiny muted";
    }
    // 显示配置文件路径
    var cfgFileEl = $("cfgCurrentFile");
    if (cfgFileEl) { cfgFileEl.textContent = t("cfgCurrentFile") + (manager.config_path || "settings.yaml"); }
  }).catch(function (e) {
    if (statusEl) { statusEl.textContent = "加载失败：" + e.message; statusEl.className = "tiny down"; }
  });
}

function setVal(id, val) {
  var el = $(id);
  if (el) { el.value = val; }
}

/* 数值读取：NaN → undefined。JSON.stringify 会省略 undefined 的键，
   后端 deep_update 因此不覆盖 yaml 原值——空输入框（loadConfig 未完成 /
   加载失败）绝不发送 null。曾因 parseInt("")=NaN → JSON 序列化成 null →
   dict 内部的 None 绕过 exclude_none → 8 个字段 pydantic 校验全挂。 */
function getNum(id) {
  var v = parseFloat(getVal(id));
  return isNaN(v) ? undefined : v;
}

function saveConfig() {
  var statusEl = $("cfgStatus");
  var btn = $("btnSaveConfig");
  if (btn) { btn.disabled = true; btn.textContent = t("cfgSaving"); }
  if (statusEl) { statusEl.textContent = "保存中…"; statusEl.className = "tiny muted"; }

  // 收集表单数据（数值走 getNum：空/非法输入 → undefined → JSON 省略该键）
  var payload = {
    webui: {
      host: getVal("cfgWebUIHost"),
      port: getNum("cfgWebUIPort"),
      reload: getVal("cfgWebUIReload") === "true"
    },
    shenji: {
      host: getVal("cfgShenjiHost"),
      port: getNum("cfgShenjiPort"),
      timeout: getNum("cfgShenjiTimeout")
    },
    mt4: {
      host: getVal("cfgMT4Host"),
      port: getNum("cfgMT4Port"),
      time_base: getVal("cfgMT4TimeBase"),
      timeout: getNum("cfgMT4Timeout"),
      poll_wait: getNum("cfgMT4PollWait"),
      drain_wait: getNum("cfgMT4DrainWait")
    },
    paths: {
      data_cache: getVal("cfgPathsDataCache"),
      artifacts: getVal("cfgPathsArtifacts"),
      tmp: getVal("cfgPathsTmp"),
      kline: getVal("cfgPathsKline"),
      shenji_db: getVal("cfgPathsShenjiDB")
    },
    data: {
      timeout: getNum("cfgDataTimeout"),
      source: getVal("cfgDataSource"),
      api_url: getVal("cfgDataAPIUrl"),
      cache_dir: getVal("cfgDataCacheDir"),
      dukascopy_user: getVal("cfgDataDukascopyUser"),
      dukascopy_password: getVal("cfgDataDukascopyPass")
    },
    logging: {
      level: getVal("cfgLoggingLevel"),
      json_lines: getVal("cfgLoggingJSON") === "true"
    },
    features: {
      enable_webui: getVal("cfgFeatureWebUI") === "true",
      enable_mt4: getVal("cfgFeatureMT4") === "true",
      enable_shenji: getVal("cfgFeatureShenji") === "true",
      enable_spec_management: getVal("cfgFeatureSpec") === "true",
      enable_backtest: getVal("cfgFeatureBacktest") === "true",
      enable_tune: getVal("cfgFeatureTune") === "true"
    }
  };

  post("/api/config", payload).then(function (res) {
    if (btn) { btn.disabled = false; btn.textContent = t("btnSaveConfig"); }
    if (statusEl) { statusEl.textContent = t("cfgSaved"); statusEl.className = "tiny ok"; }
    // 重新加载显示最新值（含环境变量覆盖后的实际生效值）
    loadConfig();
  }).catch(function (e) {
    if (btn) { btn.disabled = false; btn.textContent = t("btnSaveConfig"); }
    if (statusEl) { statusEl.textContent = t("cfgError") + e.message; statusEl.className = "tiny down"; }
  });
}

function getVal(id) {
  var el = $(id);
  return el ? el.value : "";
}

/* ── 目录选择器（File System Access API）─────────────────────────────── */
/* 仅在 localhost (secure context) 下可用；不支持时降级提示手动输入。     */
/* 注意：浏览器安全策略禁止 JS 直接读取本地绝对路径。
   选中后只能显示文件夹名称，请手动修正为完整绝对路径（如 D:\K线数据）。 */
async function pickDir(inputId) {
  var el = $(inputId);
  if (!el) return;
  if (!window.showDirectoryPicker) {
    alert("当前浏览器不支持文件夹选择器（需 Chrome 86+ / Edge 86+ 且运行在 localhost/HTTPS 环境）。\n请手动输入完整绝对路径，如 D:\\K线数据");
    return;
  }
  try {
    var dirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
    // 只能获取文件夹名，无法获取绝对路径（浏览器安全限制）
    // 显示文件夹名作为提示，用户需手动改为完整路径
    var current = el.value.trim();
    var hint = dirHandle.name;
    if (current && !current.includes(hint)) {
      // 已有不同路径，提示用户
      if (confirm("已选文件夹：" + hint + "\n当前输入：" + current + "\n\n是否替换为选中的文件夹名？\n（注意：仍需手动修正为完整绝对路径）")) {
        el.value = hint;
      }
    } else {
      el.value = hint;
    }
    // 存储 handle 供后续可能的文件操作（可选）
    el._dirHandle = dirHandle;
  } catch (e) {
    if (e.name !== "AbortError") { // 用户取消不报错
      console.error("目录选择失败:", e);
      alert("选择失败：" + e.message);
    }
  }
}

/* ── 启动 ─────────────────────────────────────────────────────────── */
function boot() {
  initTheme();
  bind();
  setView("mine");
  loadMeta();
  loadData();
  renderChart([]);
  renderCurve([]);  // 首屏无历史 → 走空状态，不画空图
  refreshLists();   // ⑧⑨ 两张表首屏就要有内容，否则一直停在"加载中…"
  loadBacktestOptions();   // 02 回测页的下拉（spec / 行情 / 画像）
  loadTuneOptions();       // 04 调优页的下拉（spec / 行情 / 历史）
  loadAcqCached();         // ①ᵇ 数据获取：本地缓存列表（来源在选定画像后加载）
  loadSpecMgmt();           // 05 Spec 管理：首屏加载全部 spec 数据
  startPolling();
}
boot();
