#include "RunConfig.h"

#include <string>
#include <vector>
#include <sstream>
#include "Config.h"

RunConfig::RunConfig(int argc, char *argv[])
{
    std::string mat1, mat2;
    mat1 = "can_24";
    mat2 = "can_24";
    if(argc == 2){
        mat1 = argv[1];
        mat2 = argv[1];
    }
    if(argc >= 3){
        mat1 = argv[1];
        mat2 = argv[2];
    }
    std::string mat1_file;
    if(mat1.find("ER") != std::string::npos){
        mat1_file = "../matrix/ER/" + mat1 +".mtx";
    }
    else if(mat1.find("G500") != std::string::npos){
        mat1_file = "../matrix/G500/" + mat1 +".mtx";
    }
    else{
        mat1_file = "../matrix/suite_sparse/" + mat1 + "/" + mat1 +".mtx";
    }
    std::string mat2_file;
    if(mat2.find("ER") != std::string::npos){
        mat2_file = "../matrix/ER/" + mat2 +".mtx";
    }
    else if(mat2.find("G500") != std::string::npos){
        mat2_file = "../matrix/G500/" + mat2 +".mtx";
    }
    else{
        mat2_file = "../matrix/suite_sparse/" + mat2 + "/" + mat2 +".mtx";
    }
    filePath = mat1_file;
    filePath2 = mat2_file;
    mat_name = mat1;
    mat_name2 = mat2;
    //printf("in RunConfig.cpp %s %s\n", filePath.c_str(), filePath2.c_str());
    printf("%s %s ", mat1.c_str(), mat2.c_str());
	Config::init("config.ini");
}


RunConfig::~RunConfig()
{
}
