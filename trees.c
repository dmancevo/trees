#include<stdio.h>
#include<omp.h>

typedef struct Node Node;

struct Node {
    int leaf;
    float val;
    int min_samples;
    int n_samples;
    int n_features;
    int * ind;
    float ** features;
    int split_ind;
    float split;
    float * g;
    Node * left;
    Node * right;
};

void fit_tree(Node * node)
{

    // Initialize left and right children.
    Node * left = malloc(sizeof(Node));
    left->min_samples=node->min_samples;
    left->n_features=node->n_features;
    left->features=node->features;
    left->g=node->g;
    
    Node * right = malloc(sizeof(Node));
    right->min_samples=node->min_samples;
    right->n_features=node->n_features;
    right->features=node->features;
    right->g=node->g;

    // Find best split.
    float best = -1, gl, gr;
    int k1, k2, nl, nr;
    for(int i=0; i<node->n_features; i++){
#pragma omp parallel private(gl, gr, k1, k2, nl, nr)
#pragma omp for
{
        for(int j=0; j<node->n_samples; j++){
            k1 = node->ind[j];
            gl=0;
            gr=0;
            for(int w=0; w<node->n_samples; w++){
                k2 = node->ind[w];
                if(node->features[i][k2] < node->features[i][k1]){
                    gl += node->g[k2];
                    nl++;
                } else {
                    gr += node->g[k2];
                    nr++;
                }
            }
#pragma omp critical
{
            if(best < gl * gl + gr * gr){
                best = gl * gl + gr * gr;
                node->split_ind = j;
                node->split = node->features[i][k1];
                left->val = gl / nl;
                right->val = gr / nr;
                left->n_samples = nl;
                right->n_samples = nr;
            }
}
        }
}
    }

    if(node->min_samples < nl && node->min_samples < nr){
        node->leaf=0;
        float ** f;
        left->ind = malloc(sizeof(f) * nl);
        right->ind = malloc(sizeof(f) * nr);
        int k, j=0, w=0;
        for(int i=0; i<node->n_samples; i++){
            k = node->ind[i];
            if(node->features[node->split_ind][k] < node->split){
                left->ind[j]=k;
                ++j;
            } else {
                left->ind[w]=k;
                ++w;
            }
        }
        node->left=left; node->right=right;
        free(node->ind);    
        // Recursive calls.
        fit_tree(right); fit_tree(left);

    } else {
        free(left);free(right);
        node->leaf=1;
    }
}

int main()
{
    return 0;
}
