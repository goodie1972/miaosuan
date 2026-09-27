//+------------------------------------------------------------------+
//| Export XAUUSD H1 to Parquet/CSV for MiaoSuan                     |
//| Save as: MQL4/Files/export_xauusd_h1_m4.mq4 (MT4)               |
//+------------------------------------------------------------------+
#property copyright "MiaoSuan"
#property link      "https://github.com/miaosuan"
#property version   "1.00"
#property strict

#include <stdlib.mqh>

input string  SymbolName   = "XAUUSD";     // 标的名称（按你的 MT4 实际品种名）
input int     TimeframeMin = 60;           // 周期分钟数（60=H1, 240=H4, 1440=D1）
input int     YearsBack    = 5;            // 往前取多少年
input string  OutputDir    = "D:\\K线数据"; // 输出目录
input bool    Overwrite    = true;         // 是否覆盖已有文件

void OnStart()
{
   // 计算起止时间
   datetime endTime = TimeCurrent();
   datetime startTime = endTime - YearsBack * 365 * 24 * 3600;
   
   // 检查品种
   if(!SymbolSelect(SymbolName, true))
   {
      Print("错误：品种不存在 ", SymbolName);
      return;
   }
   
   // 获取历史数据
   int bars = iBars(SymbolName, TimeframeMin);
   if(bars <= 0)
   {
      Print("错误：无历史数据 ", SymbolName, " ", TimeframeMin);
      return;
   }
   
   // 构造输出文件路径
   string fname = StringFormat("%s\\%s_%s_%dY.%s", 
      OutputDir, SymbolName, EnumToString((ENUM_TIMEFRAMES)TimeframeMin), YearsBack, "csv");
   
   // 打开文件
   int handle = FileOpen(fname, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("错误：无法创建文件 ", fname, " Error=", GetLastError());
      return;
   }
   
   // 写表头
   FileWrite(handle, "time,open,high,low,close,volume,tick_volume");
   
   int written = 0;
   datetime prevTime = 0;
   
   // 从旧到新遍历
   for(int i = bars - 1; i >= 0; i--)
   {
      datetime t = iTime(SymbolName, TimeframeMin, i);
      if(t < startTime) break;
      if(t == prevTime) continue; // 去重
      prevTime = t;
      
      double o = iOpen(SymbolName, TimeframeMin, i);
      double h = iHigh(SymbolName, TimeframeMin, i);
      double l = iLow(SymbolName, TimeframeMin, i);
      double c = iClose(SymbolName, TimeframeMin, i);
      long v = (long)iVolume(SymbolName, TimeframeMin, i);
      long tv = (long)iTickVolume(SymbolName, TimeframeMin, i);
      
      // Unix 秒
      FileWrite(handle, IntegerToString(t), DoubleToString(o,5), DoubleToString(h,5),
                DoubleToString(l,5), DoubleToString(c,5), LongToString(v), LongToString(tv));
      written++;
   }
   
   FileClose(handle);
   Print("完成：", fname, " 写入 ", written, " 根 K 线");
   
   // 同时生成 parquet（可选：需要 MT5 或外部转换）
   // 这里只写 CSV，Python 端一行代码可转 parquet
}