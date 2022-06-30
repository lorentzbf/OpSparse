#pragma once

#include <string>

class RunConfig
{
public:
	RunConfig(int argc, char *argv[]);
	~RunConfig();
	std::string filePath;
    std::string mat_name;
    std::string filePath2;
    std::string mat_name2;
};

