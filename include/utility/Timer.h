#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <deque>

class Timer
{
public:
    Timer()
    {
        record();
    }

    void record()
    {
        time_point_list.emplace_back(std::chrono::steady_clock::now());
    }

    void restart()
    {
        time_point_list.clear();
        record();
    }

    double costBetween(int index_e, int index_s = 0)
    {
        if (index_e >= time_point_list.size())
            index_e = time_point_list.size() - 1;
        const std::chrono::duration<double>& elapsed_seconds = time_point_list.at(index_e) - time_point_list.at(index_s);
        const double& elapsed_ms = elapsed_seconds.count() * 1000;
        return elapsed_ms;
    }

    double elapsedStart()
    {
        record();
        const double& elapsed_ms = costBetween(time_point_list.size() - 1, 0);
        return elapsed_ms;
    }

    double elapsedLast()
    {
        record();
        const double& elapsed_ms = costBetween(time_point_list.size() - 1, time_point_list.size() - 2);
        return elapsed_ms;
    }

    void checkByStart(const std::string &_about_task, const double &time_told = 0)
    {
        record();
        const double& elapsed_ms = costBetween(time_point_list.size() - 1);

        if (elapsed_ms > time_told)
        {
            std::cout.precision(3); // 10 for sec, 3 for ms
            std::cout << _about_task << ": " << elapsed_ms << " msec." << std::endl;
        }
    }

    void checkByLast(const std::string &_about_task, const double &time_told = 0)
    {
        record();
        const double& elapsed_ms = costBetween(time_point_list.size() - 1, time_point_list.size() - 2);

        if (elapsed_ms > time_told)
        {
            std::cout.precision(3); // 10 for sec, 3 for ms
            std::cout << _about_task << ": " << elapsed_ms << " msec." << std::endl;
        }
    }

private:
    std::deque<std::chrono::time_point<std::chrono::steady_clock>> time_point_list;
};
