#include "cuda_runtime.h"
#include "Executor.h"
#include "Multiply.h"
#include "DataLoader.h"
#include <iomanip>
#include "Config.h"
#include "Compare.h"
#include <cuSparseMultiply.h>
#include "Timings.h"
#include "spECKConfig.h"
#include "common.h"
#include "cuda_common.h"

template <typename DataType>
long compt_flop(const CSR<DataType> &A, const CSR<DataType> &B){
	int M = A.rows;
	long total_flop = 0;
	for(int i = 0; i < M; i++){
	    for(int j = A.row_offsets[i]; j < A.row_offsets[i+1]; j++){
	    	total_flop += B.row_offsets[A.col_ids[j]+1] - B.row_offsets[A.col_ids[j]];
	    }
	}
	return total_flop;
}


template <typename ValueType>
int Executor<ValueType>::run()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 1);
	iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	//iterationsWarmup = 1;
	//iterationsExecution = 1;
	DataLoader<ValueType> data(runConfig.filePath, runConfig.filePath2);
	//std::cout << runConfig.filePath << std::endl;
	auto& matrices = data.matrices;
	//std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	long total_flops = compt_flop(matrices.cpuA, matrices.cpuB);

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	//bool measureAll = Config::getBool(Config::TrackIndividualTimes, false);
	bool measureAll = false;
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	//bool compareData = false;
	bool compareData = true;

	if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			printf("compare data \n");
			//if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
				printf("Error: Matrix incorrect\n");
		}
		dCsrHiRes.reset();
	}

	// Multiplication
	for (int i = 0; i < iterationsExecution; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>
		(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

//		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
//		{
//			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
//				printf("Error: Matrix incorrect\n");
//		}
		dCsrHiRes.reset();
	}
	
	benchTimings /= iterationsExecution;
	benchTimings.reg_print(total_flops * 2);

	return 0;
}

template <typename ValueType>
int Executor<ValueType>::run_detail()
{
	iterationsWarmup = Config::getInt(Config::IterationsWarmUp, 1);
	iterationsExecution = Config::getInt(Config::IterationsExecution, 10);
	//iterationsWarmup = 1;
	//iterationsExecution = 1;
	DataLoader<ValueType> data(runConfig.filePath, runConfig.filePath2);
	//std::cout << runConfig.filePath << std::endl;
	auto& matrices = data.matrices;
	//std::cout << "Matrix: " << matrices.cpuA.rows << "x" << matrices.cpuA.cols << ": " << matrices.cpuA.nnz << " nonzeros\n";

	long total_flops = compt_flop(matrices.cpuA, matrices.cpuB);

	dCSR<ValueType> dCsrHiRes, dCsrReference;
	Timings timings, warmupTimings, benchTimings;
	bool measureAll = true;
	bool measureCompleteTimes = Config::getBool(Config::TrackCompleteTimes, true);
	auto config = spECK::spECKConfig::initialize(0);

	//bool compareData = false;
	bool compareData = true;

	if(Config::getBool(Config::CompareResult))
	{
		unsigned cuSubdiv_nnz = 0;
		cuSPARSE::CuSparseTest<ValueType> cusparse;
		cusparse.Multiply(matrices.gpuA, matrices.gpuB, dCsrReference, cuSubdiv_nnz);

		if(!compareData)
		{
			cudaFree(dCsrReference.data);
			dCsrReference.data = nullptr;
		}
	}

	// Warmup iterations for multiplication
	for (int i = 0; i < iterationsWarmup; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		warmupTimings += timings;

		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
		{
			printf("compare data \n");
			//if (!spECK::Compare(dCsrReference, dCsrHiRes, false))
			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
				printf("Error: Matrix incorrect\n");
		}
		dCsrHiRes.reset();
	}

	// Multiplication
	for (int i = 0; i < iterationsExecution; ++i)
	{
		timings = Timings();
		timings.measureAll = measureAll;
		timings.measureCompleteTime = measureCompleteTimes;
		spECK::MultiplyspECK<ValueType, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>
		(matrices.gpuA, matrices.gpuB, dCsrHiRes, config, timings);
		benchTimings += timings;

//		if (dCsrHiRes.data != nullptr && dCsrHiRes.col_ids != nullptr && Config::getBool(Config::CompareResult))
//		{
//			if (!spECK::Compare(dCsrReference, dCsrHiRes, compareData))
//				printf("Error: Matrix incorrect\n");
//		}
		dCsrHiRes.reset();
	}
	
	benchTimings /= iterationsExecution;
	benchTimings.print(total_flops * 2);

	return 0;
}

template class Executor<double>;
