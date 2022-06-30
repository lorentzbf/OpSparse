#ifndef Z_CSR_H_
#define Z_CSR_H_
#include <string>
#include <vector>
#include "cuda_common.h"

class CSR{
    public:
    mint M;
    mint N;
    mint nnz;
    mint *rpt;
    mint *col;
    mdouble *val;

    mint *d_rpt;
    mint *d_col;
    mdouble *d_val;
    CSR():M(0), N(0), nnz(0), 
            rpt(nullptr), col(nullptr), val(nullptr),
            d_rpt(nullptr), d_col(nullptr), d_val(nullptr)
        {}
    CSR(const std::string &mtx_file);
    CSR(const CSR& A);
    CSR(const CSR& A, mint M_, mint N_, mint M_start, mint N_start);
    ~CSR();

    void hrelease();
    void drelease();
    void release();
    void D2H();
    void H2D();
    bool operator==(const CSR& A);
    CSR& operator=(const CSR& A);
    void construct(const std::string &mtx_file);
          
};

#endif
