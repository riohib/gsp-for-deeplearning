% Test NMF on synthetic data sets 
% See Figure 4 in the paper 
clear all; close all; clc; 
% Synthetic data set experiment
m = 100; 
n = 100; 
r = 10; 
% Number of experiments 
nexp = 2; % In the paper: 50 
% Parameters for NMF algo 
timemax = Inf; 
maxiter = 500;  
% Results 
ErrAHALS = []; 
ErrNeNMF = []; 
Errl1AHALS = []; 
ErrPSNMF = []; 
ErrcPSNMF = []; 
for nex = 1 : nexp 
    clear options; 
    options.timemax = timemax; 
    options.maxiter = maxiter;
    % Generate data 
    W = max(0,randn(m,r)); 
    H = rand(r,n); 
    X = W*H; 
    nX = norm(X,'fro');  
    % Initial matrices 
    W0 = rand(m,r); 
    H0 = rand(r,n);
    % Scale initial matrices 
    alpha = sum(sum( W0'*X .* H0) ) /  sum(sum( (W0'*W0).*(H0*H0') ) ); 
    W0 = alpha*W0; 
    options.W = W0; 
    options.H = H0; 
    % Run NMF algorithms   
    % A-HALS
    [Wn,Hn,en,tn] = sparseNMF(X,r,options);
    % NeNMF
    options.NeNMF = 1; 
    [Wn2,Hn2,en2,tn2] = sparseNMF(X,r,options);
    % l1 A-HALS
    [Wl1,Hl1,el1,tl1] = sHALSacc(X,W0,H0,0.5,0.5,0.1,options.maxiter,options.timemax); 
    % PSNMF 
    options.NeNMF = 0; 
    options.sW = sp_col(W); 
    [Ws,Hs,es,ts] = sparseNMF(X,r,options); 
    % cPSNMF 
    options.colproj = 1; 
    [Wcs,Hcs,ecs,tcs] = sparseNMF(X,r,options); 
    
    % Record results 
    ErrAHALS = [ErrAHALS; en]; 
    ErrNeNMF = [ErrNeNMF; en2]; 
    Errl1AHALS = [Errl1AHALS; el1]; 
    ErrPSNMF = [ErrPSNMF; es]; 
    ErrcPSNMF = [ErrcPSNMF; ecs]; 
    
    nex  
end
% plot error 
figure; 
set(0, 'DefaultAxesFontSize', 26);
set(0, 'DefaultLineLineWidth', 2);
semilogy(mean(ErrAHALS)*100/nX,'b--'); 
hold on; 
semilogy(mean(ErrNeNMF)*100/nX,'k'); 
semilogy(mean(Errl1AHALS)*100/nX,'m-.','Linewidth',2.5); 
semilogy(100*mean(ErrPSNMF)/nX,'ro--', 'Linewidth',1); 
semilogy(100*mean(ErrcPSNMF)/nX,'g', 'Linewidth',3); 
lgd = legend('A-HALS', 'NeNMF', 'l1 A-HALS', 'PSNMF','cPSNMF'); 
lgd.FontSize = 20;
xlabel('Iterations'); 
ylabel('Relative error in %, 100||X-WH||_F/||X||_F'); 