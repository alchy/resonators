#pragma once
/*
 * core_logger.h — minimal Logger compatible with IthacaCore API.
 * Replace with the full IthacaCore core_logger.h when copying DSP files.
 */
#include <string>
#include <cstdio>

enum class LogSeverity { Debug = 0, Info, Warning, Error, Critical };

class Logger {
public:
    explicit Logger(const std::string& /*log_dir*/ = ".") {}

    void log(const char* tag, LogSeverity sev, const std::string& msg) const {
        const char* prefix[] = {"DBG","INF","WRN","ERR","CRT"};
        std::printf("[%s][%s] %s\n", prefix[(int)sev], tag, msg.c_str());
    }

    // RT-safe ring-buffer variant (stub — flushes immediately here)
    void logRT(const char* tag, LogSeverity sev, const std::string& msg) const {
        log(tag, sev, msg);
    }
};
