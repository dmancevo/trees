#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

float power(float x, int i)
{
    float y = 1;
    for(int j = 0; j < i; j++)
        y *= x;
    return y;
}

void feature_split(
        float * best_split,
        float * best_gain,
        int n_rows,
        int * ind,
        float * feature,
        float * gradient)
{
    *best_gain = -1;
    #pragma omp parallel
    {
        float gain, left, right, split;
        int k;
        #pragma omp for
        for(int i = 0; i < n_rows; i++)
        {
            left  = 0;
            right = 0;
            gain  = 0;
            split = feature[ind[i]];
            for(int j = 0; j < n_rows; j++)
            {
                k = ind[j];
                if(feature[k] < split)
                    left += gradient[k];
                else
                    right += gradient[k];
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

void split(
        int n_features,
        int n_rows,
        int * ind,
        int * best_feature,
        float * best_split,
        float ** features,
        float * gradient)
{
    float split_gain[n_features];
    float splits[n_features];
    for(int i = 0; i < n_features; i++)
        feature_split(
                &splits[i],
                &split_gain[i],
                n_rows,
                ind,
                features[i],
                gradient);

    float best_gain = 1e-7;
    for(int i = 0; i < n_features; i++)
    {
        if(best_gain < split_gain[i])
        {
            best_gain = split_gain[i];
            *best_feature = i;
            *best_split = splits[i];
        }
    }
}

typedef struct Node Node;

struct Node
{
    int feature;
    float split;
    float g;
    int leaf;
    int node_id;
    int depth;
    int * ind;
    struct Node * left;
    struct Node * right;
};

void split_node(
        struct Node * node,
        int n_features,
        int n_rows,
        float ** features,
        float * gradient)
{
    int f; float s;
    split(
        n_features,
        n_rows,
        node->ind,
        &f,
        &s,
        features,
        gradient);

    node->feature=f;
    node->split=s;

    int n_l=0, n_r=0;
    #pragma omp parallel reduction(+:n_l) reduction(+:n_r)
    {
        int l=0, r=0, k;
        #pragma omp for
        for(int i=0; i<n_rows; i++)
        {
            k = node->ind[i];
            if(features[f][k] < s)
                ++l;
            else
                ++r;
        }
        n_l += l;
        n_r += r;
    }

    if(1<n_l && 1< n_r)
    {

        int * l_ind = malloc(sizeof(int) * n_l);
        int * r_ind = malloc(sizeof(int) * n_r);
        #pragma omp parallel
        {
            int k;
            #pragma omp for
            for(int i=0; i<n_rows; i++)
            {
                k = node->ind[i];
                if(features[f][k] < s)
                    l_ind[i] = k;
                else
                    r_ind[i] = k;
            }
        }

        free(node->ind);
        node->leaf=0;

        Node L={
            .leaf=1,
            .node_id=2*node->node_id+1,
            .depth=node->depth+1,
            .ind=l_ind};
        node->left=&L;

        Node R={
            .leaf=1,
            .node_id=2*node->node_id+2,
            .depth=node->depth+1,
            .ind=r_ind};
        node->right=&R;

        // Recursive call.
        split_node(
            &L,
            n_features,
            n_l,
            features,
            gradient);
        
        split_node(
            &R,
            n_features,
            n_r,
            features,
            gradient);
    } else {
        float gr=0;
        #pragma omp parallel reduction(+:gr)
        {
            #pragma omp for
            for(int i=0; i<n_rows; i++)
            {
                gr+=gradient[node->ind[i]];
            }
        }
        node->g=gr/n_rows;
    }

}

Node tree_fit(
        int n_features,
        int n_rows,
        float ** features,
        float * gradient)
{
    // Initialize root.
    int * ind = malloc(sizeof(int) * n_rows);
    #pragma omp parallel
    {
        #pragma omp for
        for(int i=0; i<n_rows; i++)
            ind[i]=i;
    }
    Node root={
        .leaf=1,
        .node_id=0,
        .depth=0,
        .ind=ind
    };

    // Split nodes.
    split_node(
        &root,
        n_features,
        n_rows,
        features,
        gradient);
    return root;
}

void tree_predict(
        Node * root,
        int n_rows,
        float ** f,
        float * p)
{
//    #pragma omp parallel
//    {
        Node * n = root;
//        #pragma omp for
        for(int i=0; i<n_rows; i++)
        {
            while(n->leaf != 1)
            {
                if(f[n->feature][i] < n->split)
                    n = root->left;
                else
                    n = root->right;
            } 
            p[i] = n->g;
            n = root;
        }
//    }
}


int main()
{
    int n_features = 2;
    int n_rows = 8;
    int ind[8] = {0,1,2,3,4,5,6,7};
    int best_feature;
    float best_split, best_gain;
    float feature_0[8]={1,1,2,2,3,3,4,4};
    float feature_1[8]={1,1,3,3,3,3,2,2};
    float ** features;
    features=malloc(sizeof(features) * 8);
    features[0] = feature_0;
    features[1] = feature_1;
    float gradient[8]={-3,-2,-2,-2,4,4,5,5};
    Node root = tree_fit(
        n_features,
        n_rows,
        features,
        gradient);
    float * preds = malloc(8 * sizeof(float));
    tree_predict(
        &root,
        n_rows,
        features,
        preds);
    for(int i=0; i<8; i++)
        printf("pred %d %f\n",
                i,
                preds[i]);
    free(preds);
    return 0;
}
