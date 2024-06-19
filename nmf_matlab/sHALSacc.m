% Accelerated hierarchical alternating least squares (HALS) algorithm of
% Cichocki et al. with sparsity constraints 
%
% See N. Gillis and F. Glineur, "Accelerated Multiplicative Updates and 
% Hierarchical ALS Algorithms for Nonnegative Matrix Factorization”, 
% CORE Discussion Paper 2011/30. 
% See https://sites.google.com/site/nicolasgillis/ 
%
% [U,V,e,t] = sHALSacc(M,U,V,alpha,maxiter,timelimit)
%
% Input.
%   M              : (m x n) matrix to factorize
%   (U,V)          : initial matrices of dimensions (m x r) and (r x n)
%   sU             : Sparsity target for U respectively in [0,1). 
%                    (Sparsity is defined as the percentage of entries in  
%                    each column of U (resp. row of V) smaller than 1e-6  
%                    times the largest one.)
%   alpha          : nonnegative parameter of the accelerated method
%                    (alpha=0.5 seems to work well)
%   delta          : parameter to stop inner iterations when they become
%                    inneffective (delta=0.1 seems to work well). 
%   maxiter        : maximum number of iterations
%   timelimit      : maximum time alloted to the algorithm
%
% Output.
%   (U,V)    : nonnegative matrices s.t. UV approximate M
%   (e,t)    : error and time after each iteration, 
%               can be displayed with plot(t,e)
%
% Remark. With alpha = 0, it reduces to the original HALS algorithm.  

function [U,V,e,t] = sHALSacc(M,U,V,sU,alpha,delta,maxiter,timelimit)

% Initialization
etime = cputime; nM = norm(M,'fro')^2; 
[m,n] = size(M); [m,r] = size(U);
a = 0; e = []; t = []; iter = 0; 

if nargin <= 4, alpha = 0.5; end
if nargin <= 5, delta = 0.1; end
if nargin <= 6, maxiter = 100; end
if nargin <= 7, timelimit = 60; end

% Normalization of the columns of U and V so that max(U(:,i)) = 1
d = max(U); U = U*diag(d.^-1); V = diag(d)*V; 
% Scaling 
eit1 = cputime; A = M*V'; B = V*V'; eit1 = cputime-eit1; j = 0;
scaling = sum(sum(A.*U))/sum(sum( B.*(U'*U) )); 
U = U*scaling; 
d = max(U); 
U = U*diag(d.^-1); 
V = diag(d)*V; 
    
% Penalty parameters for columns of U
lambdaU = nM/sum(U(:))*0.05*ones(r,1)*sU; 

% Main loop
while iter <= maxiter && cputime-etime <= timelimit
    % Update of U
    if j == 1, % Do not recompute A and B at first pass
        % Use actual computational time instead of estimates rhoU
        eit1 = cputime; A = M*V'; B = V*V'; eit1 = cputime-eit1; 
    end
    j = 1; eit2 = cputime; eps = 1; eps0 = 1;
    % Update of U 
    U = sHALSupdt(M',U',B',A',eit1,alpha,delta,lambdaU); U = U';
    % Normalization of the columns of U and V so that max(U(:,i)) = 1
    d = max(U); 
    U = U*diag(d.^-1); 
    V = diag(d)*V; 
    % Current sparsity of U 
    for ku = 1 : r
        sUcurrent(ku) = sp_col(U(:,ku)); 
    end
    % Update sparsity parameters 
    if mean(sUcurrent) < 0.999*sU
        % Update penalty parameters for U depending on current iterate
        % Increase if sparsity is too low
        inddim = find(sUcurrent < 0.999*sU); 
        lambdaU(inddim) = lambdaU(inddim)*1.01; 
        % Decrease if sparsity is too large
        indaug = find(sUcurrent > 1.05*sU); 
        lambdaU(indaug) = lambdaU(indaug)*0.99; 
    elseif mean(sUcurrent) > 1.001*sU
        % Decrease if sparsity is too high
        indaug = find(sUcurrent > 1.001*sU); 
        lambdaU(indaug) = lambdaU(indaug)*0.99; 
        % Increase if sparsity is too low
        inddim = find(sUcurrent < 0.95*sU); 
        lambdaU(inddim) = lambdaU(inddim)*1.01; 
    end
       
    % Update of V
    eit1 = cputime; A = (U'*M); B = (U'*U); eit1 = cputime-eit1;
    eit2 = cputime; eps = 1; eps0 = 1; 
    V = HALSupdt(M,V,B,A,eit1,alpha,delta); 
    
    % Evaluation of the error e at time t
    if nargout >= 3
        cnT = cputime;
        e = [e sqrt( (nM-2*sum(sum(V.*A))+ sum(sum(B.*(V*V')))) )]; 
        etime = etime+(cputime-cnT);
        t = [t cputime-etime];
    end
    iter = iter + 1; 
end

% Update of V <- HALS(M,U,V)
% i.e., optimizing min_{V >= 0} ||M-UV||_F^2 
% with an exact block-coordinate descent scheme
function V = HALSupdt(M,V,UtU,UtM,eit1,alpha,delta)
[r,n] = size(V); 
eit2 = cputime; % Use actual computational time instead of estimates rhoU
cnt = 1; % Enter the loop at least once
eps = 1; eps0 = 1; eit3 = 0;
while cnt == 1 || (cputime-eit2 < (eit1+eit3)*alpha && eps >= (delta)^2*eps0)
    nodelta = 0; if cnt == 1, eit3 = cputime; end
        for k = 1 : r
            deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
            V(k,:) = V(k,:) + deltaV;
            nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
            if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end % safety procedure
        end
    if cnt == 1
        eps0 = nodelta; 
        eit3 = cputime-eit3; 
    end
    eps = nodelta; cnt = 0; 
end

% Update of V <- sHALS(M,U,V)
% i.e., optimizing min_{V >= 0} ||M-UV||_F^2 s.t. ||V(i,:)||_inf = 1. 
% with an exact block-coordinate descent scheme
function V = sHALSupdt(M,V,UtU,UtM,eit1,alpha,delta,lambda)
[r,n] = size(V); 
eit2 = cputime; % Use actual computational time instead of estimates rhoU
cnt = 1; % Enter the loop at least once
eps = 1; eps0 = 1; eit3 = 0;
while (cnt == 1 || (cputime-eit2 < (eit1+eit3)*alpha && eps >= (delta)^2*eps0)) && cnt <= 10 
    nodelta = 0; if cnt == 1, eit3 = cputime; end
        for k = 1 : r
            Vold = V(k,:); 
            deltaV = max((UtM(k,:)-UtU(k,:)*V-lambda(k))/UtU(k,k),-V(k,:));
            V(k,:) = V(k,:) + deltaV;
            nodelta = nodelta + deltaV*deltaV'; % used to compute norm(V0-V,'fro')^2;
            if V(k,:) == 0, 
                % safety procedure:
                V(k,:) = Vold + 0.5*deltaV;
            end 
            V(k,:) = V(k,:)/max(V(k,:)); 
        end
    if cnt == 1
        eps0 = nodelta; 
        eit3 = cputime-eit3; 
    end
    eps = nodelta; 
    cnt = cnt + 1; 
end