#include <iostream>
#include <chrono>

inline void printProgressBar(float progress, float total) {
    float ratio = progress / total;
    int barWidth = 70;
    int barLength = static_cast<int>(ratio * barWidth);

    std::cout << "[";
    for (int i = 0; i < barLength; ++i)
    {
        std::cout << "=";
    }
    for (int i = barLength; i < barWidth; ++i)
    {
        std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(ratio * 100.0) << "% ";
    std::cout.flush();

    static auto start_time = std::chrono::steady_clock::now();
    if (progress < 1e-6)
        start_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

    float remaining_ratio = 1.0 - ratio;
    int estimated_remaining_time = static_cast<int>(elapsed_time / ratio * remaining_ratio);

    std::cout << "Current msg duration: " << progress << "s Elapsed: " << elapsed_time << "s Remaining: " << estimated_remaining_time << "s\r";
    std::cout.flush();
}
