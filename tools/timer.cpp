#include "timer.h"

void timer::start(){startTime = clock();}
void timer::end(){endTime = clock();}
void timer::result(){
    time = (double)(endTime - startTime);
    printf("unknown function : %.3f s, %.3f ms\n", time / CLOCKS_PER_SEC, time);
}
void timer::result(const char* name){
    time = (double)(endTime - startTime);
    printf("%s : %.3f s, %.3f ms\n", name, time / CLOCKS_PER_SEC, time);
}