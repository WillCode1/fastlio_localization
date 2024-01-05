#pragma once
#include <cstdio>
#include <chrono>
#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
using namespace std;

class TimeStamp {
public:
    static const char *GetLocalTimeStamp() {
            static char buf[50];

            auto now_st = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            int now_msec = now_st % 1000;
            auto now_sec = now_st / 1000;

            struct tm local_t;
#ifdef _WIN32
        localtime_s(&local_t, &now_sec);
#else
        localtime_r(&now_sec, &local_t);
#endif
        snprintf(buf, 50, "[%04d-%02d-%02d %02d:%02d:%02d %03d]: ",
                 (1900 + local_t.tm_year), (1 + local_t.tm_mon), local_t.tm_mday, 
                 local_t.tm_hour, local_t.tm_min, local_t.tm_sec, now_msec);
        return buf;
    }
};

enum LogLevel
{
        debug,
        info,
        warn,
        error,
        fatal
};

// #define LOG_LEVEL (debug)
#define LOG_LEVEL (info)

#define ENABLE_LOG (0)
#define ENABLE_LOG (1)
const std::string location_log_file = "/home/ant/location.log";

#define LOG_PRINT(level, color, ...)                                  \
    do                                                                \
    {                                                                 \
        if (level >= LOG_LEVEL)                                       \
        {                                                             \
                printf(color);                                        \
                printf("[%s]", #level);                               \
                printf("%s", TimeStamp::GetLocalTimeStamp());         \
                if (level >= error)                                   \
                        printf("(%s, %d), ", fs::path(__FILE__).filename().string().c_str(), __LINE__); \
                printf(__VA_ARGS__);                                  \
                printf("\033[0m\n");                                  \
                if (ENABLE_LOG)                                         \
                {                                                       \
                        FILE *location_log = fopen(location_log_file.c_str(), "a");      \
                        fprintf(location_log, "[%s]", #level);                               \
                        fprintf(location_log, "%s", TimeStamp::GetLocalTimeStamp());         \
                        if (level >= error)                                   \
                                fprintf(location_log, "(%s, %d), ", fs::path(__FILE__).filename().string().c_str(), __LINE__); \
                        fprintf(location_log, __VA_ARGS__);                                  \
                        fprintf(location_log, "\n");                                  \
                        fclose(location_log);                           \
                }                                                       \
        }                                                             \
    } while (0)

#define LOG_DEBUG(...) LOG_PRINT(debug, "\033[0;32m", __VA_ARGS__)
#define LOG_INFO(...) LOG_PRINT(info, "\033[0m", __VA_ARGS__)
#define LOG_WARN(...) LOG_PRINT(warn, "\033[1;33m", __VA_ARGS__)
#define LOG_ERROR(...) LOG_PRINT(error, "\033[1;31m", __VA_ARGS__)
#define LOG_FATAL(...) LOG_PRINT(fatal, "\033[1;31m", __VA_ARGS__)

#define LOG_DEBUG_COND(cond, ...) if (cond) LOG_DEBUG(__VA_ARGS__)
#define LOG_INFO_COND(cond, ...) if (cond) LOG_INFO(__VA_ARGS__)
#define LOG_WARN_COND(cond, ...) if (cond) LOG_WARN(__VA_ARGS__)
#define LOG_ERROR_COND(cond, ...) if (cond) LOG_ERROR(__VA_ARGS__)
#define LOG_FATAL_COND(cond, ...) if (cond) LOG_FATAL(__VA_ARGS__)
