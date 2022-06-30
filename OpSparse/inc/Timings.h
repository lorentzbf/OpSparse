#ifndef __Z_TIMING_H__
#define __Z_TIMING_H__

class Timings {
    public:
    bool measure_separate;
    bool measure_total;
    double setup;
    double symbolic_binning;
    double symbolic;
    double reduce;
    double numeric_binning;
    double prefix;
    double allocate;
    double numeric;
    double cleanup;
    double total;
    Timings();

    void operator+=(const Timings& b);

    void operator/=(const double x);
    void print(const double total_flop);
    void reg_print(const double total_flop);
    void perf_print(const double total_flop);
    void binning_print(const double total_flop);
};

#endif

