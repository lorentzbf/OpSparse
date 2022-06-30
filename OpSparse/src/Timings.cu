#include "Timings.h"
#include <stdio.h>

Timings::Timings(){
    measure_separate = true;
    measure_total = true;
    setup = 0;
    symbolic_binning = 0;
    symbolic = 0;
    reduce = 0;
    numeric_binning = 0;
    prefix = 0;
    allocate = 0;
    numeric = 0;
    cleanup = 0;
    total = 0;
}

void Timings::operator+=(const Timings& b){
    setup += b.setup;
    symbolic_binning += b.symbolic_binning;
    symbolic += b.symbolic;
    reduce += b.reduce;
    numeric_binning += b.numeric_binning;
    prefix += b.prefix;
    allocate += b.allocate;
    numeric += b.numeric;
    cleanup += b.cleanup;
    total += b.total;
}

void Timings::operator/=(const double x){
    setup /= x;
    symbolic_binning /= x;
    symbolic /= x;
    reduce /= x;
    numeric_binning /= x;
    prefix /= x;
    allocate /= x;
    numeric /= x;
    cleanup /= x;
    total /= x;
}

void Timings::print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("total flop %lf\n", total_flop);
    double sum_total = setup + symbolic_binning + symbolic + numeric_binning
        + reduce + prefix + allocate + numeric + cleanup;
    if(measure_separate){
        //printf("time(ms): setup %.3lf symbolic_binning %.3lf symbolic %.3lf numeric_binning %.3lf prefix_allocate %.3lf numeric %.3lf cleanup %.3lf total %.3lf",)
        printf("time(ms):\n");
        printf("    setup            %8.3lfms %6.2lf%%\n", 1000*setup, setup/total*100);
        printf("\e[1;31m    symbolic_binning %8.3lfms %6.2lf%%\n\e[0m", 1000*symbolic_binning, symbolic_binning/total*100);
        printf("\e[1;31m    symbolic         %8.3lfms %6.2lf%%\n\e[0m", 1000*symbolic, symbolic/total*100);
        printf("    reduce            %8.3lfms %6.2lf%%\n", 1000*reduce, reduce/total*100);
        printf("\e[1;31m    numeric_binning  %8.3lfms %6.2lf%%\n\e[0m", 1000*numeric_binning, numeric_binning/total*100);
        printf("    prefix           %8.3lfms %6.2lf%%\n", 1000*prefix, prefix/total*100);
        printf("    allocate         %8.3lfms %6.2lf%%\n", 1000*allocate, allocate/total*100);
        printf("\e[1;31m    numeric          %8.3lfms %6.2lf%%\n\e[0m", 1000*numeric, numeric/total*100);
        printf("    cleanup          %8.3lfms %6.2lf%%\n", 1000*cleanup, cleanup/total*100);
        printf("    sum_total        %8.3lfms %6.2lf%%\n", 1000*sum_total, sum_total/total*100);
        printf("    total            %8.3lfms %6.2lf%%\n", 1000*total, total/total*100);
        printf("perf(Gflops):\n");
        printf("    setup            %6.2lf\n", total_flop_G/setup);
        printf("    symbolic_binning %6.2lf\n", total_flop_G/symbolic_binning);
        printf("    symbolic         %6.2lf\n", total_flop_G/symbolic);
        printf("    reduce           %6.2lf\n", total_flop_G/reduce);
        printf("    numeric_binning  %6.2lf\n", total_flop_G/numeric_binning);
        printf("    prefix           %6.2lf\n", total_flop_G/prefix);
        printf("    allocate         %6.2lf\n", total_flop_G/allocate);
        printf("    numeric          %6.2lf\n", total_flop_G/numeric);
        printf("    cleanup          %6.2lf\n", total_flop_G/cleanup);
        printf("    total            %6.2lf\n", total_flop_G/total);
    }
}
        
void Timings::reg_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("%6.2lf\n", total_flop_G/total);
}

void Timings::perf_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    printf("%6.2lf %6.2lf\n", total_flop_G/symbolic, total_flop_G/numeric);
}

void Timings::binning_print(double total_flop){
    double total_flop_G = total_flop/1000000000;
    double total_binning_time = symbolic_binning + numeric_binning;
    printf("%.4le %.4lf\n", total_binning_time, 100*total_binning_time/total);
}
