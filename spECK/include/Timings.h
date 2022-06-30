#pragma once
#include <vector>

class Timings {
public:
    bool measureAll;
    bool measureCompleteTime;
    float init;
    float countProducts;
    float loadBalanceCounting;
    float globalMapsCounting;
    float spGEMMCounting;
    float allocC;
    float loadBalanceNumeric;
    float globalMapsNumeric;
    float spGEMMNumeric;
    float sorting;
    float cleanup;
    float complete;

    float setup;
    float symbolic_binning;
    float symbolic;
    float numeric_binning;
    float prefix;
    float allocate;
    float numeric;
    float total;
    
    Timings(){
        measureAll = false;
        measureCompleteTime = false;
        init = 0.0f;
        countProducts = 0.0f;
        loadBalanceCounting = 0.0f;
        globalMapsCounting = 0.0f;
        spGEMMCounting = 0.0f;
        allocC = 0.0f;
        loadBalanceNumeric = 0.0f;
        globalMapsNumeric = 0.0f;
        spGEMMNumeric = 0.0f;
        sorting = 0.0f;
        cleanup = 0.0f;
        complete = 0.0f;
    }
    void operator+=(const Timings& b) {
        init += b.init;
        countProducts += b.countProducts;
        loadBalanceCounting += b.loadBalanceCounting;
        globalMapsCounting += b.globalMapsCounting;
        spGEMMCounting += b.spGEMMCounting;
        allocC += b.allocC;
        loadBalanceNumeric += b.loadBalanceNumeric;
        globalMapsNumeric += b.globalMapsNumeric;
        spGEMMNumeric += b.spGEMMNumeric;
        sorting += b.sorting;
        cleanup += b.cleanup;
        complete += b.complete;
    }

    void operator/=(const float& x) {
        init /= x;
        countProducts /= x;
        loadBalanceCounting /= x;
        globalMapsCounting /= x;
        spGEMMCounting /= x;
        allocC /= x;
        loadBalanceNumeric /= x;
        globalMapsNumeric /= x;
        spGEMMNumeric /= x;
        sorting /= x;
        cleanup /= x;
        complete /= x;
    }
    void print(long long total_flop){
        float total_flop_d = float(total_flop)/1000000;
        setup = init + countProducts;
        symbolic_binning = loadBalanceCounting;
        symbolic = globalMapsCounting + spGEMMCounting;
        numeric_binning = loadBalanceNumeric;
        prefix = 0;
        allocate = allocC;
        numeric = globalMapsNumeric + spGEMMNumeric + sorting;
        total = complete;

        //if (measureAll){
            printf("spECK     initial mallocs = %f ms\n", init);
            printf("spECK  count computations = %f ms\n", countProducts);
            printf("spECK       load-balancer = %f ms\n", loadBalanceCounting);
            printf("spECK      GlobalMaps Cnt = %f ms\n", globalMapsCounting);
            printf("spECK     counting kernel = %f ms\n", spGEMMCounting);
            printf("spECK        malloc mat C = %f ms\n", allocC);
            printf("spECK   num load-balancer = %f ms\n", loadBalanceNumeric);
            printf("spECK     init GlobalMaps = %f ms\n", globalMapsNumeric);
            printf("spECK      numeric kernel = %f ms\n", spGEMMNumeric);
            printf("spECK      Sorting kernel = %f ms\n", sorting);
            printf("spECK             cleanup = %f ms\n", cleanup);
            printf("--------------------------------------------------------------\n");
        //}

        //if(measureAll){
            printf("time(ms):\n");
            printf("    setup            %8.3lfms %6.2lf%%\n", setup, setup/total*100);
            printf("\e[1;31m    symbolic_binning %8.3lfms %6.2lf%%\n\e[0m", symbolic_binning, symbolic_binning/total*100);
            printf("\e[1;31m    symbolic         %8.3lfms %6.2lf%%\n\e[0m", symbolic, symbolic/total*100);
            printf("\e[1;31m    numeric_binning  %8.3lfms %6.2lf%%\n\e[0m", numeric_binning, numeric_binning/total*100);
            printf("    prefix           %8.3lfms %6.2lf%%\n", prefix, prefix/total*100);
            printf("    allocate         %8.3lfms %6.2lf%%\n", allocate, allocate/total*100);
            printf("\e[1;31m    numeric          %8.3lfms %6.2lf%%\n\e[0m", numeric, numeric/total*100);
            printf("    cleanup          %8.3lfms %6.2lf%%\n", cleanup, cleanup/total*100);
            printf("    total            %8.3lfms %6.2lf%%\n", total, total/total*100);
            printf("perf(Gflops):\n");
            printf("    setup            %6.2lf\n", total_flop_d/setup);
            printf("    symbolic_binning %6.2lf\n", total_flop_d/symbolic_binning);
            printf("    symbolic         %6.2lf\n", total_flop_d/symbolic);
            printf("    numeric_binning  %6.2lf\n", total_flop_d/numeric_binning);
            printf("    prefix           %6.2lf\n", total_flop_d/prefix);
            printf("    allocate         %6.2lf\n", total_flop_d/allocate);
            printf("    numeric          %6.2lf\n", total_flop_d/numeric);
            printf("    cleanup          %6.2lf\n", total_flop_d/cleanup);
            printf("    total            %6.2lf\n", total_flop_d/total);
        //}
    }
    void reg_print(long long total_flop){
        float total_flop_d = float(total_flop)/1000000;
        total = complete;
        printf("%6.2lf\n", total_flop_d/total);
    }
    void binning_print(long long total_flop){
        float total_flop_d = float(total_flop)/1000000;
        float total_binning_time = loadBalanceCounting + loadBalanceNumeric;
        printf("%.4e %.4f\n", total_binning_time/1000, 100*total_binning_time/complete);
    }

};


