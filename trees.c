#include<stdio.h>
#include<omp.h>

typedef struct Node Node;

struct Node {
    int leaf;
    float val;
    int min_samples;
    int split_ind;
    float split;
    Node * left;
    Node * right;
};

void fit_tree(
    Node * node,
    int n_features,
    int n_samples,
    int * ind,
    float ** features,
    float * gradient)
{

    // Initialize left and right children.
    Node * left = malloc(sizeof(Node));
    left->min_samples=node->min_samples;
    
    Node * right = malloc(sizeof(Node));
    right->min_samples=node->min_samples;

    // Find best split.
    float best = -1, gl, gr;
    int nl, nr, left_n, right_n;
    for(int f=0; f<n_features; f++){
        for(int i=0; i<n_samples; i++){
            gl=0;
            gr=0;
            for(int j=0; j<n_samples; j++){
                if(features[f][ind[j]] < features[f][ind[i]]){
                    gl += gradient[ind[j]];
                    nl++;
                } else {
                    gr += gradient[ind[j]];
                    nr++;
                }
            }
            if(best < gl * gl + gr * gr){
                best = gl * gl + gr * gr;
                node->split_ind = f;
                printf("f=%d i=%d ind[i]=%d\n", f, i, ind[i]);
                printf("features[f][ind[i]]=%f\n", features[f][ind[i]]);
                node->split = features[f][ind[i]];
//                left->val = gl / nl;
//                right->val = gr / nr;
//                left_n = nl;
//                right_n = nr;
            }
        }
    }
//
//    if(node->min_samples < nl && node->min_samples < nr){
//        node->leaf=0;
//        left->ind = malloc(sizeof(int) * nl);
//        right->ind = malloc(sizeof(int) * nr);
//        int k, j=0, w=0;
//        for(int i=0; i<node->n_samples; i++){
//            k = node->ind[i];
//            if(node->features[node->split_ind][k] < node->split){
//                left->ind[j]=k;
//                ++j;
//            } else {
//                left->ind[w]=k;
//                ++w;
//            }
//        }
//        node->left=left; node->right=right;
//        free(node->ind);    
//      // Recursive calls.
//        fit_tree(right); fit_tree(left);
//
//    } else {
//        free(left);free(right);
//        node->leaf=1;
//    }
}

int main()
{
    float ** features;
    features = malloc(sizeof(features));
    float feature_0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float feature_1[8] = {1, 1, 1, 1, 1, 1, 2, 2};
    features[0] = feature_0;
    features[1] = feature_1;

    float gradient[8] = {1,1,1,1,-1,-1,-1,-1};
    int * ind = malloc(8 * sizeof(int));
    for(int i=0; i<8; i++) 
        ind[i] = i;

    Node * root = malloc(sizeof(root));
    root->leaf=1;
    root->min_samples=1;
    
    fit_tree(root, 2, 8, ind, features, gradient);
    return 0;
}
