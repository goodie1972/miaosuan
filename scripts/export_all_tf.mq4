//+------------------------------------------------------------------+
//| Export XAUUSD All Timeframes to CSV for MiaoSuan                |
//| 保存到: MT4 数据目录/MQL4/Scripts/export_all_tf.mq4            |
//| 编译后在 MT4 脚本面板运行即可                                   |
//+------------------------------------------------------------------+
#property copyright "MiaoSuan"
#property version   "1.10"
#property strict
#property show_inputs   // 运行时弹出参数面板

input string  InpSymbol     = "XAUUSD";      // 标的（按你 MT4 实际品种名，如 XAUUSDm）
input int     InpYearsBack  = 5;             // 往前取多少年
input string  InpOutputDir  = "D:\\K线数据";  // 输出目录
input bool    InpDoM15      = true;          // 导出 M15
input bool    InpDoM30      = true;          // 导出 M30
input bool    InpDoH1       = true;          // 导出 H1
input bool    InpDoH4       = true;          // 导出 H4
input bool    InpDoD1       = true;          // 导出 D1

//+------------------------------------------------------------------+
//| 导出单个周期                                                      |
//+------------------------------------------------------------------+
int ExportTF(string symbol, int tf_min, string tf_name, int years_back, string out_dir)
{
   // 计算时间范围
   datetime endTime   = TimeCurrent();
   datetime startTime = endTime - (datetime)(years_back * 365 * 24 * 3600);

   // 检查品种可用性
   if(!SymbolSelect(symbol, true))
   {
      Print("跳过 ", symbol, " ", tf_name, "：品种不可用");
      return 0;
   }

   // 确保输出目录存在
   FolderCreate(out_dir);

   // 构造文件名
   string fname = StringFormat("%s\\%s_%s_%dY.csv",
      out_dir, symbol, tf_name, years_back);

   int handle = FileOpen(fname, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("错误：无法创建文件 ", fname, " Error=", GetLastError());
      return 0;
   }

   // 写表头
   FileWrite(handle, "time,open,high,low,close,volume,tick_volume");

   int bars = iBars(symbol, tf_min);
   if(bars <= 0)
   {
      Print("错误：无历史数据 ", symbol, " ", tf_name);
      FileClose(handle);
      return 0;
   }

   int written = 0;
   datetime prevTime = 0;

   // 从旧到新遍历
   for(int i = bars - 1; i >= 0; i--)
   {
      datetime t = iTime(symbol, tf_min, i);
      if(t < startTime) break;
      if(t == prevTime) continue;
      prevTime = t;

      double o = iOpen(symbol, tf_min, i);
      double h = iHigh(symbol, tf_min, i);
      double l = iLow(symbol, tf_min, i);
      double c = iClose(symbol, tf_min, i);
      long   v = (long)iVolume(symbol, tf_min, i);
      long  tv = (long)iTickVolume(symbol, tf_min, i);

      FileWrite(handle, IntegerToString(t),
                DoubleToString(o, 5),
                DoubleToString(h, 5),
                DoubleToString(l, 5),
                DoubleToString(c, 5),
                LongToString(v),
                LongToString(tv));
      written++;
   }

   FileClose(handle);
   Print("完成：", fname, " 写入 ", written, " 根 K 线");
   return written;
}

//+------------------------------------------------------------------+
//| 脚本入口                                                          |
//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== MiaoSuan 批量导出开始 ===");
   Print("品种: ", InpSymbol, "  年数: ", InpYearsBack);
   Print("输出目录: ", InpOutputDir);

   int total = 0;

   if(InpDoM15) total += ExportTF(InpSymbol, PERIOD_M15, "M15", InpYearsBack, InpOutputDir);
   if(InpDoM30) total += ExportTF(InpSymbol, PERIOD_M30, "M30", InpYearsBack, InpOutputDir);
   if(InpDoH1)  total += ExportTF(InpSymbol, PERIOD_H1,  "H1",  InpYearsBack, InpOutputDir);
   if(InpDoH4)  total += ExportTF(InpSymbol, PERIOD_H4,  "H4",  InpYearsBack, InpOutputDir);
   if(InpDoD1)  total += ExportTF(InpSymbol, PERIOD_D1,  "D1",  InpYearsBack, InpOutputDir);

   Print("=== 全部完成！共 ", total, " 根 K 线 ===");
   Print("下一步：运行 scripts/convert_all.py 转 parquet");
}