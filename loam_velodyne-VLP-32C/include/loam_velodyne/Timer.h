#ifndef TIMER_H
#define TIMER_H
#include <time.h>
#include <string>

/*
Profiling 단순 타이머.
사용법 : 
    1. timer 선언
    2. 시작과 끝에 timer.start(), timer.stop()
    3. 평균 필요 시 timer.time(micro, milli, sec) / iteration.
*/ 
// Ver1
/*
class timer{
private:
    struct timespec begin, end;
    long time;
public:
    void start(){
        clock_gettime(CLOCK_MONOTONIC, &begin);}
    void stop(){
        clock_gettime(CLOCK_MONOTONIC, &end);
        time = (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec);};
    double timeMicro(){return (double)time/1000;}
    double timeMilli(){return (double)time/1000000;}
    double timeSec(){return (double)time/1000000000;}
    void printSec(std::string name){printf("%s : %lf secs\n", name, timeSec());}
}; 
*/
// Ver2
/*
*/
namespace loam{
class Timer{
private:
    clock_t startTime, endTime;
    double time;
public:
    explicit Timer();
    ~Timer();
    void start();
    void end();
    void result();
    void result(const char* name);
};
}
#endif //TIMER_H
