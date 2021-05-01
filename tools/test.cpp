#include <stdio.h>
#include "timer.h"
#include <string>
#include <iostream>

// int main(void){
//     timer t = timer();

//     double temp = 0;
//     t.start();
//     for(int i = 0;i<1000000;i++){
//         for(int j = 0;j<1000;j++){
//             temp *= i;
//             temp /= i;
//         }
//     }
//     t.stop();

//     printf("%lf secs\n", t.timeSec());
//     return 0;
// }

int main(){
    timer t;
    clock_t start, end;
    double result;
    start = clock();
    t.start();
    int temp = 0;
    for(int i = 0; i<1000000;i++){
        for(int j = 0;j<5000;j++){
            temp += j;
        }
    }
    end = clock();
    t.end();
    result = (double)(end-start);
    std::cout << "result : " << ((result) / CLOCKS_PER_SEC) << " seconds\n" << end;
    printf("%f\n", result / CLOCKS_PER_SEC);
    t.result("hello ");

}