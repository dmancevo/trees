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
    #pragma omp parallel private(nl, nr, gl, gr)
    #pragma omp for
    {
        for(int i=0; i<n_samples; i++){
            gl=0; gr=0;
            nl=0; nr=0;
            for(int j=0; j<n_samples; j++){
                if(features[f][ind[j]] < features[f][ind[i]]){
                    gl += gradient[ind[j]];
                    nl++;
                } else {
                    gr += gradient[ind[j]];
                    nr++;
                }
            }
            #pragma omp critical
            {if(best < gl * gl + gr * gr){
                best = gl * gl + gr * gr;
                node->split_ind = f;
                node->split = features[f][ind[i]];
                left->val = gl / nl;
                right->val = gr / nr;
                left_n = nl;
                right_n = nr;
            }}
        }
    }}

    if(    node->min_samples <= left_n
        && node->min_samples <= right_n
        && 0 < left_n && 0 < right_n){

        node->leaf=0;
        int * left_ind = malloc(sizeof(int) * nl);
        int * right_ind = malloc(sizeof(int) * nr);
        int k, j=0, w=0;
        for(int i=0; i<n_samples; i++){
            k = ind[i];
            if(features[node->split_ind][k] < node->split){
                left_ind[j]=k;
                ++j;
            } else {
                right_ind[w]=k;
                ++w;
            }
        }
        node->left=left; node->right=right;
        free(ind);

        // Recursive calls.
        fit_tree(left, n_features, left_n,
            left_ind, features, gradient);
        fit_tree(right, n_features, right_n,
            right_ind, features, gradient);

    } else {
        free(left);free(right);
        node->leaf=1;
    }
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

    Node * root = malloc(sizeof(Node));
    root->leaf=1;
    root->min_samples=1;

    fit_tree(root, 2, 8, ind, features, gradient);

    printf("root.split=%f root.leaf=%d\n",
        root->split, root->leaf);
    printf("root.left.split=%f root.left.leaf=%d root.left.val=%f\n",
        root->left->split, root->left->leaf, root->left->val);
    printf("root.right.split=%f root.right.leaf=%d root.right.val=%f\n",
        root->right->split, root->right->leaf, root->right->val);

    return 0;
}
