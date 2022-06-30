#include "DataLoader.h"

#include <iostream>
#include "COO.h"
#include <cuSparseMultiply.h>

template<typename T>
std::string typeExtension();
template<>
std::string typeExtension<float>()
{
	return std::string("");
}
template<>
std::string typeExtension<double>()
{
	return std::string("d_");
}

template class DataLoader<float>;
template class DataLoader<double>;

template <typename ValueType>
DataLoader<ValueType>::DataLoader(std::string path, std::string path2) : matrices()
{
	std::string csrPath = path + typeExtension<ValueType>() + ".hicsr";

	try
	{
		//std::cout << "trying to load csr file \"" << csrPath << "\"\n";
		matrices.cpuA = loadCSR<ValueType>(csrPath.c_str());
		//std::cout << "successfully loaded: \"" << csrPath << "\"\n";
	}
	catch (std::exception& ex)
	{
		//std::cout << "could not load csr file:\n\t" << ex.what() << "\n";
		try
		{
			//std::cout << "trying to load mtx file \"" << path << "\"\n";
			COO<ValueType> cooMat = loadMTX<ValueType>(path.c_str());
			convert(matrices.cpuA, cooMat);
			//std::cout << "successfully loaded and converted: \"" << csrPath << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			std::cout << "could not load mtx file: \"" << path << "\"\n";
			throw "could not load mtx file";
		}

		try
		{
			//std::cout << "write csr file for future use in" << csrPath.c_str() << "\n";
			//storeCSR(matrices.cpuA, csrPath.c_str());
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
		}
	}
	
	//cuSPARSE::CuSparseTest<ValueType> cuSparse;
	
	//calculate the transpose if matrix is not square
    if(path == path2){
		convert(matrices.cpuB, matrices.cpuA, 0);
    }
    else{
        try
		{
			//std::cout << "trying to load mtx file \"" << path << "\"\n";
			COO<ValueType> cooMat = loadMTX<ValueType>(path2.c_str());
			convert(matrices.cpuB, cooMat);
			//std::cout << "successfully loaded and converted: \"" << csrPath << "\"\n";
		}
		catch (std::exception& ex)
		{
			std::cout << ex.what() << std::endl;
			std::cout << "could not load mtx file: \"" << path << "\"\n";
			throw "could not load mtx file";
		}
        if(matrices.cpuA.cols == matrices.cpuB.rows){
            // do nothing
        }
        else if(matrices.cpuA.cols < matrices.cpuB.rows){
            CSR<ValueType> tmp(matrices.cpuB, matrices.cpuA.cols, matrices.cpuB.cols, 0, 0);
            matrices.cpuB = tmp;
        }
        else{
            CSR<ValueType> tmp(matrices.cpuA, matrices.cpuA.rows, matrices.cpuB.rows, 0, 0);
            matrices.cpuA = tmp;
        }
    }

	//if (matrices.gpuA.rows != matrices.gpuA.cols)
	//{
	//	cuSparse.Transpose(matrices.gpuA, matrices.gpuB);
	//	convert(matrices.cpuB, matrices.gpuB);
	//}
	//else
	//{
	//	convert(matrices.gpuB, matrices.cpuA, 0);
	//	convert(matrices.cpuB, matrices.cpuA, 0);
	//}
	convert(matrices.gpuA, matrices.cpuA, 0);
	convert(matrices.gpuB, matrices.cpuB, 0);
}
