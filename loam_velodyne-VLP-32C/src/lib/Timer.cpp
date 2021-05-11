#include "loam_velodyne/Timer.h"
namespace loam{
	/*
	 * Constructor
	 * 
	 */
    Timer::Timer(){
		this->ResetCallTime();
		time_total_ = 0;
		call_time_threshold_ = 1000000;
	}

    void Timer::Start(){
		start_time_ = clock();
	}
    void Timer::End(){
		end_time_ = clock();
		call_time_++;
		time_total_ = time_total_ + (double)(end_time_ - start_time_);
	}
    void Timer::Result(){
        time_ = (double)(end_time_ - start_time_);
        printf("unknown function : %.3f s, %.3f ms\n", time_ / CLOCKS_PER_SEC, time_);
    }
    void Timer::Result(const char* name){
        time_ = (double)(end_time_ - start_time_);
        printf("%s : %.3f s, %.3f ms\n", name, time_ / CLOCKS_PER_SEC, time_);
    }
	int Timer::GetCallTime(){
		return call_time_;
	}
	void Timer::ResetCallTime(){
		call_time_ = 0;
	}
	void Timer::ResultBySec(const char* name){
		printf("hello\n");
	}
	void Timer::ResultByCallTime(const char* name){
		if(call_time_%call_time_threshold_==0){
			//printf("%d :: call_time_, %0.3f :: time_total\n",call_time_, time_total_);
			printf("=========ResultByCallTime=========\n");
			this.Result(name);
			printf("%.3f s, %.3f ms per 1 calls\n\n\n", (time_total_/call_time_threshold_)/CLOCKS_PER_SEC, time_total_/call_time_threshold_);
		time_total_ = 0;
		}
	}

	
	Timer::~Timer(){}

}
