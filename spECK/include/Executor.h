#pragma once
#include "RunConfig.h"

template<typename ValueType>
class Executor
{
public:
	Executor(int argc, char *argv[]) : runConfig(argc, argv) {}
	~Executor() = default;
	int run();
	int run_detail();

private:
	RunConfig runConfig;
	int iterationsWarmup = 0;
	int iterationsExecution = 1;
};

