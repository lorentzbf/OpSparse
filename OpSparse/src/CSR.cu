#include <cassert>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "cuda_common.h"
#include "CSR.h"


#define SYMMETRY_GENERAL 0
#define SYMMETRY_SYMMETRY 1
#define SYMMETRY_SKEW_SYMMETRY 2
#define SYMMETRY_HERMITIAN 3
struct matrix_market_banner
{
    std::string matrix; // "matrix" or "vector"
    std::string storage;    // "array" or "coordinate", storage_format
    std::string type;       // "complex", "real", "integer", or "pattern"
    std::string symmetry;   // "general", "symmetric", "hermitian", or "skew-symmetric"
};

inline void tokenize(std::vector<std::string>& tokens, const std::string str, const std::string delimiters = "\n\r\t ")
{
    tokens.clear();
    // Skip delimiters at beginning.
    std::string::size_type first_pos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type last_pos     = str.find_first_of(delimiters, first_pos);

    while (std::string::npos != first_pos || std::string::npos != last_pos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(first_pos, last_pos - first_pos));
        // Skip delimiters.  Note the "not_of"
        first_pos = str.find_first_not_of(delimiters, last_pos);
        // Find next "non-delimiter"
        last_pos = str.find_first_of(delimiters, first_pos);
    }
}

template <typename Stream>
void read_mm_banner(Stream& input, matrix_market_banner& banner)
{
    std::string line;
    std::vector<std::string> tokens;

    // read first line
    std::getline(input, line);
    tokenize(tokens, line);

    if (tokens.size() != 5 || tokens[0] != "%%MatrixMarket" || tokens[1] != "matrix")
        throw std::runtime_error("invalid MatrixMarket banner");

    banner.matrix = tokens[1]; // mow just matrix, no vector
    banner.storage  = tokens[2]; // now just coordinate(sparse), no array(dense)
    banner.type     = tokens[3]; // int, real, pattern for double, complex for two double
    banner.symmetry = tokens[4]; // general, symmetry, etc

    if(banner.matrix != "matrix" && banner.matrix != "vector")
        throw std::runtime_error("invalid MatrixMarket matrix type: " + banner.matrix);
    if(banner.matrix == "vector")
        throw std::runtime_error("not impl matrix type: " + banner.matrix);

    if (banner.storage != "array" && banner.storage != "coordinate")
        throw std::runtime_error("invalid MatrixMarket storage format [" + banner.storage + "]");
    if(banner.storage == "array")
        throw std::runtime_error("not impl storage type "+ banner.storage);

    if (banner.type != "complex" && banner.type != "real" && banner.type != "integer" && banner.type != "pattern")
        throw std::runtime_error("invalid MatrixMarket data type [" + banner.type + "]");
    //if(banner.type == "complex")
    //    throw std::runtime_error("not impl data type: " + banner.type);

    if (banner.symmetry != "general" && banner.symmetry != "symmetric" && banner.symmetry != "hermitian" && banner.symmetry != "skew-symmetric")
        throw std::runtime_error("invalid MatrixMarket symmetry [" + banner.symmetry + "]");
    if(banner.symmetry == "hermitian")
        throw std::runtime_error("not impl matrix type: " + banner.symmetry);
    
}

template <typename index_type, typename value_type>
class Pair{
    public:
    index_type ind;
    value_type val;
    friend bool operator<=(const Pair &lhs, const Pair& rhs){
        return lhs.ind <= rhs.ind;
    }
    friend bool operator<(const Pair &lhs, const Pair& rhs){
        return lhs.ind < rhs.ind;
    }
    friend bool operator>(const Pair &lhs, const Pair& rhs){
        return lhs.ind > rhs.ind;
    }

};

void CSR::hrelease(){
    delete [] rpt;
    rpt = nullptr;
    delete [] col;
    col = nullptr;
    delete [] val;
    val = nullptr;
}

void CSR::drelease(){
    CHECK_ERROR(cudaFree(d_rpt));
    d_rpt = nullptr;
    //CHECK_ERROR(cudaFree(d_combined));
    //d_combined = nullptr;
    CHECK_ERROR(cudaFree(d_col));
    CHECK_ERROR(cudaFree(d_val));
    d_col = nullptr;
    d_val = nullptr;
}

void CSR::release(){
    hrelease();
    drelease();
}

CSR::~CSR(){
    release();
}

void CSR::H2D(){
    drelease();
    CHECK_ERROR(cudaMalloc(&d_rpt, (M+1)*sizeof(mint)));
    //mint aligned_nnz = (nnz + 1) >> 1 << 1;
    //CHECK_ERROR(cudaMalloc(&d_combined, aligned_nnz * (sizeof(mint) + sizeof(mdouble))));
    //d_col = (mint*)d_combined;
    //d_val = (mdouble*)(d_col + aligned_nnz);
    CHECK_ERROR(cudaMalloc(&d_col, nnz*sizeof(mint)));
    CHECK_ERROR(cudaMalloc(&d_val, nnz*sizeof(mdouble)));
    CHECK_ERROR(cudaMemcpy(d_rpt, rpt, (M+1)*sizeof(mint), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_col, col, nnz*sizeof(mint), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_val, val, nnz*sizeof(mdouble), cudaMemcpyHostToDevice));
}

void CSR::D2H(){
    hrelease();
    rpt = new mint [M+1];
    col = new mint [nnz];
    val = new mdouble [nnz];
    CHECK_ERROR(cudaMemcpy(rpt, d_rpt, (M+1)*sizeof(mint), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(col, d_col, nnz*sizeof(mint), cudaMemcpyDeviceToHost));
    CHECK_ERROR(cudaMemcpy(val, d_val, nnz*sizeof(mdouble), cudaMemcpyDeviceToHost));
}

CSR::CSR(const CSR &A){
    //printf("construct A\n");
    M = A.M;
    N = A.N;
    nnz = A.nnz;
    rpt = new mint [M+1];
    col = new mint [nnz];
    val = new double [nnz];
    memcpy(rpt, A.rpt, (M+1)*sizeof(mint));
    memcpy(col, A.col, nnz*sizeof(mint));
    memcpy(val, A.val, nnz*sizeof(mdouble));
    d_rpt = nullptr;
    d_col = nullptr;
    d_val = nullptr;
}

CSR& CSR::operator=(const CSR &A){
    //printf("construct A\n");
    M = A.M;
    N = A.N;
    nnz = A.nnz;
    rpt = new mint [M+1];
    col = new mint [nnz];
    val = new double [nnz];
    memcpy(rpt, A.rpt, (M+1)*sizeof(mint));
    memcpy(col, A.col, nnz*sizeof(mint));
    memcpy(val, A.val, nnz*sizeof(mdouble));
    d_rpt = nullptr;
    d_col = nullptr;
    d_val = nullptr;
    return *this;
}

CSR::CSR(const CSR &A, mint M_, mint N_, mint M_start = 0, mint N_start = 0){
    assert(M_ + M_start <= A.M && "matrix subsect error M");
    assert(N_ + N_start <= A.N && "matrix subsect error N");
    mint M_end = M_start + M_;
    mint N_end = N_start + N_;
    M = M_;
    N = N_;
    mint *row_size = new mint [M];
    memset(row_size, 0, M*sizeof(mint));
    for(mint i = M_start; i < M_end; i++){
        for(mint j = A.rpt[i]; j < A.rpt[i+1]; j++){
            if(A.col[j]>= N_start && A.col[j] < N_end){
                row_size[i - M_start]++;
            }
        }
    }

    rpt = new mint [M+1];
    rpt[0] = 0;
    for(mint i = 0; i < M; i++){
        rpt[i+1] = rpt[i] + row_size[i];
    }
    nnz = rpt[M];
    delete [] row_size;

    col = new mint [nnz];
    val = new mdouble [nnz];
    for(mint i = M_start; i < M_end; i++){
        mint jj = rpt[i - M_start];
        for(mint j = A.rpt[i]; j < A.rpt[i+1]; j++){
            if(A.col[j]>= N_start && A.col[j] < N_end){
                col[jj] = A.col[j] - N_start;
                val[jj++] = A.val[j];
            }
        }
    }
    d_rpt = nullptr;
    d_col = nullptr;
    d_val = nullptr;
    //d_combined = nullptr;
}


bool CSR::operator==(const CSR &rhs){
    if(nnz != rhs.nnz){
        printf("nnz not equal %d %d\n", nnz, rhs.nnz);
        throw std::runtime_error("nnz not equal");
    }
    assert(M == rhs.M && "dimension not same");
    assert(N == rhs.N && "dimension not same");
    //assert(nnz == rhs.nnz && "dimension not same");
    int error_num = 0;
    double epsilon = 1e-9;
    for(mint i = 0; i < M; i++){
        if(unlikely(error_num > 10))
            throw std::runtime_error("matrix compare: error num exceed threshold");
        if(unlikely(rpt[i] != rhs.rpt[i])){
            printf("rpt not equal at %d rows, %d != %d\n", i, rpt[i], rhs.rpt[i]);
            error_num++;
        }
        for(mint j = rpt[i]; j < rpt[i+1]; j++){
            if(unlikely(error_num > 10))
                throw std::runtime_error("matrix compare: error num exceed threshold");
            if(col[j] != rhs.col[j]){
                printf("col not equal at %d rows, index %d != %d\n", i, col[j], rhs.col[j]);
                error_num++;
            }
            if(!(std::fabs(val[j] - rhs.val[j]) < epsilon || 
            std::fabs(val[j] - rhs.val[j]) < epsilon * std::fabs(val[j]))){
                printf("val not eqaul at %d rows, value %.18le != %.18le\n", i, val[j], rhs.val[j]);
                error_num++;
            }
        }
    }
    if(rpt[M] != rhs.rpt[M]){
        printf("rpt[M] not equal\n");
        throw std::runtime_error("matrix compare: error num exceed threshold");
    }
    if(error_num)
        return false;
    else
        return true;
}

CSR::CSR(const std::string &mtx_file){
    construct(mtx_file);
}

void CSR::construct(const std::string &mtx_file){
    d_rpt = nullptr;
    d_col = nullptr;
    d_val = nullptr;
    std::ifstream ifile(mtx_file.c_str());
    if(!ifile){
        throw std::runtime_error(std::string("unable to open file \"") + mtx_file + std::string("\" for reading"));
    }
    matrix_market_banner banner;
    // read mtx header
    read_mm_banner(ifile, banner);

    // read file contents line by line
    std::string line;

    // skip over banner and comments
    do
    {
        std::getline(ifile, line);
    } while (line[0] == '%');

    // line contains [num_rows num_columns num_entries]
    std::vector<std::string> tokens;
    tokenize(tokens, line);

    if (tokens.size() != 3)
        throw std::runtime_error("invalid MatrixMarket coordinate format");

    std::istringstream(tokens[0]) >> M;
    std::istringstream(tokens[1]) >> N;
    std::istringstream(tokens[2]) >> nnz;
    assert(nnz > 0 && "something wrong: nnz is 0");

    mint *I_ = new mint [nnz];
    mint *J_ = new mint [nnz];
    mdouble *coo_values_ = new mdouble [nnz];

    mint num_entries_read = 0;

    // read file contents
    if (banner.type == "pattern")
    {
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            num_entries_read++;
        }
        std::fill(coo_values_, coo_values_ + nnz, mdouble(1));
    }
    else if (banner.type == "real" || banner.type == "integer")
    {
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read];
            num_entries_read++;
        }
    }
    else if (banner.type == "complex")
    {
        mdouble tmp;
        while(num_entries_read < nnz && !ifile.eof())
        {
            ifile >> I_[num_entries_read];
            ifile >> J_[num_entries_read];
            ifile >> coo_values_[num_entries_read] >> tmp;
            num_entries_read++;
        }
    }
    else
    {
        throw std::runtime_error("invalid MatrixMarket data type");
    }
    ifile.close();

    if(num_entries_read != nnz)
        throw std::runtime_error("read nnz not equal to decalred nnz " + std::to_string(num_entries_read));

    //// check validity of row and column index data
    //size_t min_row_index = *std::min_element(I_, I_+nnz);
    //size_t max_row_index = *std::max_element(I_, I_+nnz);
    //size_t min_col_index = *std::min_element(J_, J_+nnz);
    //size_t max_col_index = *std::max_element(J_, J_+nnz);

    //if (min_row_index < 1)
    //    throw std::runtime_error("found invalid row index (index < 1)");
    //if (min_col_index < 1)
    //    throw std::runtime_error("found invalid column index (index < 1)");
    //if (max_row_index > M)
    //    throw std::runtime_error("found invalid row index (index > num_rows)");
    //if (max_col_index > N)
    //    throw std::runtime_error("found invalid column index (index > num_columns)");

    // convert base-1 indices to base-0
    for(mint n = 0; n < nnz; n++){
        I_[n] -= 1;
        J_[n] -= 1;
    }

    // expand symmetric formats to "general" format
    if (banner.symmetry != "general"){
        mint non_diagonals = 0;

        for (mint n = 0; n < nnz; n++)
            if(likely(I_[n] != J_[n]))
                non_diagonals++;

        mint new_nnz = nnz + non_diagonals;

        mint* new_I = new mint [new_nnz];
        mint* new_J = new mint [new_nnz];
        mdouble *new_coo_values;
        new_coo_values = new mdouble [new_nnz];
        

        if (banner.symmetry == "symmetric"){
            mint cnt = 0;
            for (mint n = 0; n < nnz; n++){
                // copy entry over
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;

                // duplicate off-diagonals
                if (I_[n] != J_[n]){
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        }
        else if (banner.symmetry == "skew-symmetric"){
            mint cnt = 0;
            for (mint n = 0; n < nnz; n++){
                // copy entry over
                new_I[cnt] = I_[n];
                new_J[cnt] = J_[n];
                new_coo_values[cnt] = coo_values_[n];
                cnt++;

                // duplicate off-diagonals
                if (I_[n] != J_[n]){
                    new_I[cnt] = J_[n];
                    new_J[cnt] = I_[n];
                    new_coo_values[cnt] = -coo_values_[n];
                    cnt++;
                }
            }
            assert(new_nnz == cnt && "something wrong: new_nnz != cnt");
        }
        else if (banner.symmetry == "hermitian"){
            // TODO
            throw std::runtime_error("MatrixMarket I/O does not currently support hermitian matrices");
        }

        // store full matrix in coo
        nnz = new_nnz;
        delete [] I_;
        delete [] J_;
        delete [] coo_values_;
        I_ = new_I;
        J_ = new_J;
        coo_values_ = new_coo_values;
    } // if (banner.symmetry != "general")

    // sort indices by (row,column)
    Pair<long, mdouble> *p = new Pair<long, mdouble> [nnz];
    for(mint i = 0; i < nnz; i++){
        p[i].ind = (long int)N * I_[i] + J_[i];
        p[i].val = coo_values_[i];
    }
    std::sort(p, p + nnz);
    for(mint i = 0; i < nnz; i++){
        I_[i] = p[i].ind / N;
        J_[i] = p[i].ind % N;
        coo_values_[i] = p[i].val;
    }
    delete [] p;
    
    // coo to csr
    rpt = new mint [M+1];
    memset(rpt, 0, (M + 1) * sizeof(mint));
    for(mint i = 0; i < nnz; i++){
        rpt[I_[i]+1]++;
    }
    for(mint i = 1; i <= M; i++){
        rpt[i] += rpt[i-1];
    }
    delete [] I_;
    col = J_;
    val = coo_values_;

    // check csr format
    assert(rpt[0] == 0 && "first row_pointer != 0");
    for(mint i = 0; i < M; i++){
        if(likely(rpt[i]<= rpt[i+1] && rpt[i] <= nnz)){
            for(mint j = rpt[i]; j < rpt[i+1] - 1; j++){
                if(likely(col[j] < col[j+1])){}
                else{
                	printf("row %d, col_index %d, index %d\n", i, col[j], j);
                    throw std::runtime_error("csr col_index not in assending order");
                }
            }
            for(mint j = rpt[i]; j < rpt[i+1]; j++){
                if(likely(col[j] < N && col[j] >= 0)){}
                else{
                	printf("row %d, col_index %d, index %d\n", i, col[j], j);
                    throw std::runtime_error("csr col_index out of range");
                }
            }
        }
        else{
            printf("i %d  row_pointer[i] %d row_pointer[i+1] %d\n", i, rpt[i], rpt[i+1]);
            throw std::runtime_error("csr row_pointer wrong");
        }
    }
    assert(rpt[M] == nnz && "last row_pointer != nnz_");

}


