This code allows you to compute grouped sparse projections as described in the paper 

Explicit Group Sparse Projection with Applications to Deep Learning and NMF

by Riyasat Ohib, Nicolas Gillis, Niccolo Dalmasso, Sameena Shah, Vamsi K. Potluru, Sergey Plis, 2022.

The main function is weightedgroupedsparseproj.m that allows you to perform such a projection. 
The function weightedgroupedsparseproj_col.m can be used more conveniently to project the columns of a matrix so that they have a target average sparsity level. 
************Example************
>> X = rand(4,6)

X =

    0.8147    0.6324    0.9575    0.9572    0.4218    0.6557
    0.9058    0.0975    0.9649    0.4854    0.9157    0.0357
    0.1270    0.2785    0.1576    0.8003    0.7922    0.8491
    0.9134    0.5469    0.9706    0.1419    0.9595    0.9340

>> weightedgroupedsparseproj_col(X,0.5)

ans =

    0.6646    0.6324    0.9460    1.0705         0    0.1704
    0.9495         0    0.9657         0    0.9337         0
         0         0         0    0.5972    0.5657    0.7874
    0.9732         0    0.9809         0    1.0641    1.0581
********************************


This code can also be used to compute *sparse NMFs*: Given a nonnegative matrix X (m-by-n) and r, find two noonnegative matrices W (m-by-r) and H (r-by-n) such that WH approximates X, and where W and/or H are sparse. 
You can run the examples from the paper using test_sNMF_CBCL.m (on the CBCL data set) and test_sNMF_synth.m (on the synthetic data sets). 

************Example************
>> X = rand(7,6)

X =

    0.3491    0.3064    0.9706    0.1959    0.6061    0.7242
    0.9812    0.4496    0.3580    0.2880    0.5374    0.6911
    0.2963    0.4205    0.2427    0.7341    0.4093    0.0846
    0.2104    0.9581    0.5303    0.4823    0.8455    0.8041
    0.3628    0.5713    0.6259    0.6766    0.6152    0.2925
    0.7631    0.5570    0.1948    0.5148    0.8180    0.9610
    0.4404    0.3313    0.4359    0.4483    0.2425    0.3368

>> options.sW = 0.75; options.maxiter = 30; r = 3; 
>> [W,H,e,t] = sparseNMF(X,r,options);
Iteration number and error:
10:1.136 - 20:1.136 - 30:1.136 - 
>> W

W =

         0    0.7483         0
    1.2127         0         0
         0    0.4697         0
         0    0.9615         0
         0    0.7286         0
    0.0624    0.8738         0
         0         0    0.9649

>> H

H =

    0.8209    0.3658    0.2786    0.2354    0.4446    0.5797
    0.4965    0.7438    0.6538    0.6285    0.8660    0.7922
    0.4564    0.3433    0.4518    0.4646    0.2513    0.3491********************************
