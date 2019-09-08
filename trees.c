#include<stdio.h>
#include<omp.h>

typedef struct Node Node;
struct Node {
    int leaf;
    float g;
    int min_samples;
    int split_ind;
    float split;
    Node * left;
    Node * right;
};

Node * get_root(int min_samples)
{
    Node * root = malloc(sizeof(Node));
    root->leaf=1;
    root->min_samples=min_samples;
    return root;
}

void fit_tree(
    Node * root,
    int n_features,
    int n_samples,
    int * ind,
    float ** features,
    float * gradient)
{

    // Initialize left and right children.
    Node * left = malloc(sizeof(Node));
    left->min_samples=root->min_samples;

    Node * right = malloc(sizeof(Node));
    right->min_samples=root->min_samples;

    // Find best split.
    float best = -1, gl, gr;
    int nl, nr, left_n, right_n;
    for(int f=0; f<n_features; f++){
    #pragma omp parallel private(nl, nr, gl, gr)
    {
        #pragma omp for
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
                root->split_ind = f;
                root->split = features[f][ind[i]];
                root->g = (gl + gr) / (nl + nr);
                left->g = gl / nl;
                right->g = gr / nr;
                left_n = nl;
                right_n = nr;
            }}
        }
    }}

    if(    root->min_samples <= left_n
        && root->min_samples <= right_n
        && 0 < left_n && 0 < right_n){

        root->leaf=0;
        int * left_ind = malloc(sizeof(int) * nl);
        int * right_ind = malloc(sizeof(int) * nr);
        int k, j=0, w=0;
        for(int i=0; i<n_samples; i++){
            k = ind[i];
            if(features[root->split_ind][k] < root->split){
                left_ind[j]=k;
                ++j;
            } else {
                right_ind[w]=k;
                ++w;
            }
        }
        root->left=left; root->right=right;
        free(ind);

        // Recursive calls.
        fit_tree(left, n_features, left_n,
            left_ind, features, gradient);
        fit_tree(right, n_features, right_n,
            right_ind, features, gradient);

    } else {
        free(left);free(right);
        root->leaf=1;
    }
}

void tree_predict(
    Node * root,
    int n_samples,
    float ** features,
    float * predictions)
{
    Node * node;
    #pragma omp parallel private(node)
    {
    #pragma omp for
    for(int i=0; i<n_samples; i++){
        node = root;
        while(node->leaf != 1){
            if(features[node->split_ind][i] < node->split)
                node = node->left;
            else
                node = node->right;
        }
        predictions[i] = node->g;
    }}
}

typedef struct GBM GBM;
struct GBM {
    int left;
    float learning_rate;
    GBM * next;
    Node * root;
};

GBM * get_gbm(int n_trees, float learning_rate){
    GBM * gbm = malloc(sizeof(GBM));
    gbm->left=n_trees-1;
    gbm->learning_rate=learning_rate;
    return gbm;
}

void fit_gbm(
    GBM * gbm,
    int min_samples,
    int n_features,
    int n_samples,
    float ** features,
    float * labels){

    GBM * curr = gbm;
    float * predictions = malloc(sizeof(float) * n_samples);
    float * gbm_predictions = malloc(sizeof(float) * n_samples);
    float * gradient = malloc(sizeof(float) * n_samples);
    #pragma omp parallel
    {
    #pragma omp for
    for(int i=0; i<n_samples; i++){
        gbm_predictions[i] = 0;
        gradient[i] = labels[i];
    }}

    while(0 <= curr->left){

        curr->root = get_root(min_samples);

        int * ind = malloc(sizeof(int) * n_samples);
        for(int i=0; i<n_samples; i++)
            ind[i] = i;

        fit_tree(curr->root, n_features, n_samples,
                    ind, features, gradient);

        tree_predict(curr->root, n_samples,
                        features, predictions);

        curr->next = malloc(sizeof(GBM));
        curr->next->left = curr->left-1;
        curr->next->learning_rate = curr->learning_rate;
        curr = curr->next;

        for(int i=0; i<n_samples; i++){
            gbm_predictions[i] += predictions[i];
            gradient[i] = labels[i] - gbm_predictions[i];
        }
    }

    for(int i=0; i<n_samples; i++)
        printf("label[i]=%f gbm_pred[i]=%f\n", labels[i], gbm_predictions[i]);

    free(predictions);
    free(gbm_predictions);
    free(gradient);
}

int main()
{
    float ** features;
    features = malloc(sizeof(features));
    float feature_0[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float feature_1[8] = {1, 1, 1, 1, 1, 1, 2, 2};
    features[0] = feature_0;
    features[1] = feature_1;

    float labels[8] = {1,1,1,1,-1,-1,-1,-1};

    GBM * gbm = get_gbm(10, 1.0);
    fit_gbm(gbm, 1, 2, 8, features, labels);

    return 0;
}
