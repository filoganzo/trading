//+------------------------------------------------------------------+
//|  FiloHFT/Logger.mqh                                              |
//|  Plaintext diagnostic logger. Distinct from Telemetry — this is  |
//|  human-readable operational log (warnings, errors, lifecycle).   |
//|  High-volume structured event streams go through Telemetry.mqh.  |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_LOGGER_MQH__
#define __FILOHFT_LOGGER_MQH__

#include <FiloHFT/Common.mqh>

enum ENUM_LOG_LEVEL {
   LOG_DEBUG = 0,
   LOG_INFO  = 1,
   LOG_WARN  = 2,
   LOG_ERROR = 3
};

string LogLevelName(const ENUM_LOG_LEVEL l) {
   switch(l) {
      case LOG_DEBUG: return "DEBUG";
      case LOG_INFO:  return "INFO";
      case LOG_WARN:  return "WARN";
      case LOG_ERROR: return "ERROR";
   }
   return "?";
}

class Logger {
private:
   int             m_file;
   ENUM_LOG_LEVEL  m_min_level;
   string          m_path;
   bool            m_to_terminal;

public:
   void Init(const string path, ENUM_LOG_LEVEL min_level = LOG_INFO, bool to_terminal = true) {
      m_path        = path;
      m_min_level   = min_level;
      m_to_terminal = to_terminal;
      m_file        = FileOpen(path, FILE_WRITE | FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
      if(m_file != INVALID_HANDLE) {
         FileSeek(m_file, 0, SEEK_END);
      } else {
         PrintFormat("[Logger] could not open %s (err=%d)", path, GetLastError());
      }
   }

   void Close() {
      if(m_file != INVALID_HANDLE) { FileClose(m_file); m_file = INVALID_HANDLE; }
   }

   void Log(ENUM_LOG_LEVEL level, const string component, const string message) {
      if(level < m_min_level) return;
      string ts = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);
      string line = StringFormat("%s [%s] %s: %s", ts, LogLevelName(level), component, message);
      if(m_to_terminal) Print(line);
      if(m_file != INVALID_HANDLE) {
         FileWriteString(m_file, line + "\n");
         FileFlush(m_file);
      }
   }

   void Debug(const string c, const string m) { Log(LOG_DEBUG, c, m); }
   void Info (const string c, const string m) { Log(LOG_INFO,  c, m); }
   void Warn (const string c, const string m) { Log(LOG_WARN,  c, m); }
   void Error(const string c, const string m) { Log(LOG_ERROR, c, m); }
};

//--- singleton instance
Logger g_log;

#endif // __FILOHFT_LOGGER_MQH__
