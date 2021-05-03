#include "loam_velodyne/Timer.h"
namespace loam{
    Timer::Timer(){}
    Timer::~Timer(){}
    void Timer::start(){startTime = clock();}
    void Timer::end(){endTime = clock();}
    void Timer::result(){
        time = (double)(endTime - startTime);
        printf("unknown function : %.3f s, %.3f ms\n", time / CLOCKS_PER_SEC, time);
    }
    void Timer::result(const char* name){
        time = (double)(endTime - startTime);
        printf("%s : %.3f s, %.3f ms\n", name, time / CLOCKS_PER_SEC, time);
    }
}
