% Compare A-HALS with PSNMF on the CBCL data set
% See Figure 5 and 6 in the paper 
clear all; close all; clc;
load CBCL.mat 
r = 49;
[m,n] = size(X);
nX = norm(X,'fro');
options.timemax = Inf;
options.maxiter = 500;
W0 = rand(m,r);
H0 = rand(r,n);
% Scale
alpha = sum(sum( W0'*X .* H0) ) /  sum(sum( (W0'*W0).*(H0*H0') ) );
W0 = alpha*W0;
options.W = W0;
options.H = H0;
% Run NMF algorithms
% A-HALS
disp('***A-HALS***')
[Wn,Hn,en,tn] = sparseNMF(X,r,options);
% PSNMF with sparsity of A-HALS
disp('***PSNMF***')
options.sW = sp_col(Wn);
[Ws,Hs,es,ts] = sparseNMF(X,r,options);
% PSNMF with sparsity 0.85
disp('***PSNMF 0.85***')
options.sW = 0.85;
[Ws2,Hs2,es2,ts2] = sparseNMF(X,r,options);
% plot error
figure;
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
plot(en*100/nX,'b');
hold on;
plot(100*es/nX,'r-.', 'Linewidth',1);
plot(100*es2/nX,'r', 'Linewidth',1);
legend(sprintf('A-HALS (sparsity(W)=%2.2f)',sp_col(Wn)),... 
       sprintf('PSNMF (sparsity(W)=%2.2f)',sp_col(Ws)),... 
       sprintf('PSNMF (sparsity(W)=%2.2f)',sp_col(Ws2))); 
xlabel('Iterations');
ylabel('100||X-WH||_F/||X||_F');
% Display the facials features 
affichage(Wn,7,19,19); title('A-HALS');
affichage(Ws,7,19,19); title('PSNMF');
affichage(Ws2,7,19,19); title('PSNMF 0.85');