#include <stdio.h>
#include <omp.h>

void feature_split(
        int length,
        float * best_split,
        float * best_gain,
        float * feature,
        float * gradient)
{
    *best_gain = -1;
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
                if(*best_gain < gain)
                {
                    *best_gain  = gain;
                    *best_split = split;
                }
            }
        }
    }
}

void split(int n_features,
        float ** features,
        float * gradient)
{
    float split_gain[n_features];
    float split_feature[n_features];
    for(int i = 0; i < n_features; i++)
        feature_split(n_rows,
                &split_feature[i],
                &split_gain[i],
                features[i],
                gradient);

    float best_gain = split_gain[0];
    int feature = 0;
    for(int i = 1; i < n_features; i++)
    {
        if(best_gain < split_gain[i])
        {
            best_gain = split_gain[i];
            feature = i;
        }
    }
}

int main()
{
    float best_split, best_gain;
    float feature[8] = {1,2,3,4,5,6,7,8};
    float gradient[8] = {-1,-1,-1,-1,1,1,1,1};
    feature_split(8, &best_split, &best_gain,
            feature, gradient);
    printf("best_split=%f best_gain=%f\n",
            best_split, best_gain);
    return 0;
}
