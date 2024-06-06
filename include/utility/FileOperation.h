#pragma once
#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

class FileOperation
{
public:
    static void createFileWhenNotExist(const std::string &path)
    {
        if (!fs::exists(path))
        {
            std::ofstream file(path);
            file.close();
        }
    }

    static bool createDirectoryOrRecreate(const std::string &directory_path, bool exist_remove = true)
    {
        if (fs::exists(directory_path))
        {
            if (exist_remove)
                fs::remove_all(directory_path);
            else
                return true;
        }

        return fs::create_directories(directory_path);
    }

    static int getFilesNumByExtension(const std::string &directory_path, const std::string& extension)
    {
        int scd_file_count = 0;
        if (!fs::exists(directory_path))
            return -1;

        for (const auto &entry : fs::directory_iterator(directory_path))
        {
            if (fs::is_regular_file(entry))
            {
                if (entry.path().extension() == extension)
                    scd_file_count++;
            }
        }
        return scd_file_count;
    }

    static std::string getOneFilenameByExtension(const std::string &directory_path, const std::string& extension)
    {
        if (!fs::exists(directory_path))
            return std::string("");

        for (const auto &entry : fs::directory_iterator(directory_path))
        {
            if (fs::is_regular_file(entry))
            {
                if (entry.path().extension() == extension)
                {
                    return entry.path().filename().string();
                }
            }
        }
        return std::string("");
    }
};
