#include <stdio.h>
#include <omp.h>

float split(int length, float * feature, float * gradient)
{
    float best_split, best_gain = -1;
    #pragma omp parallel
    {
        float gain, left, right, split;
        #pragma omp for
        for(int i = 0; i < length; i++)
        {
            left  = 0;
            right = 0;
            gain  = 0;
            split = feature[i];
            for(int j = 0; j < length; j++)
            {
                if(feature[j] < split)
                    left += gradient[j];
                else
                    right += gradient[j];
            }
            gain = right * right + left * left;
            #pragma omp critical
            {
                if(best_gain < gain)
                {
                    best_gain  = gain;
                    best_split = split;
                }
            }
        }
    }
    return best_split;
}

int main()
{
    float feature[8] = {1,2,3,4,5,6,7,8};
    float gradient[8] = {-1,-1,-1,-1,1,1,1,1};
    printf("split=%f\n", split(8, feature, gradient));
    return 0;
}
