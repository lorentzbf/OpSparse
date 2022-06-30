#ifdef _WIN32
#include <intrin.h>
//surpress crash notification windows (close or debug program window)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <x86intrin.h>
#endif
#include <string>
#include "Executor.h"

int main(int argc, char *argv[])
{
#ifdef _WIN32
	//surpress crash notification windows (close or debug program window)
	SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX);
#endif
	Executor<double> exe(argc, argv);
	return exe.run();
}
