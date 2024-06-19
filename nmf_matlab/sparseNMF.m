% Sparse NMF using fast gradient and coordinate descent 
% 
% min_{W>=0,H>=0} || X - WH ||_F such that the average sparsity of the 
% columns of W is given by s. 
%
% ******
% input  
% ******
% X      : m-byn matrix to factorize
% r      : rand of the factorization 
% s      : average sparsity of the columns of W, the first factor in the
%          factorization. For s=[], no sparsity constraint enforced. 
% options: 
%          - sW       : average sparsity of the columns of W, the first 
%                       factor in the factorization. For sW=[] (default), no 
%                       sparsity constraint enforced. 
%          - sH       : average sparsity of the rows of H, the second  
%                       factor in the factorization. For sH=[] (default), no 
%                       sparsity constraint enforced. 
%          - maxiter  : maximum number of iterations (number of times W and H
%                       are updated). -default=100
%          - timemax  : maximum alloted time 
%          - delta    : stopping criterion for inner iterations  -default=0.1
%          - inneriter: maximum number of inner iterations when updating W
%                       and H. -default=10
%          - W and H  : initial matrices. default-rand(m,r) and rand(r,n) 
%                       +scaling
%          - w        : weights w{i} i=1,2,...,r to compute the projection
%                       -default=/, meaning w{i}=ones(m,1) for all i
%                       (standard Hoyer sparsity). 
%          - NeNMF    : Update of H using FGM if options.NeNMF = 1
%                       (default:0)
%
% ******
% output 
% ******
% W (m by r) and H (r by n) such that ||WH-W||_F is small and the averge
% sparsity of the columns of W is s. 
%
% (e,t): error and time, plot(t,e) plots the error ||X-WH||_F over time
%
% See the paper Grouped sparse projection by Nicolas Gillis, Riyasat Ohib, 
% Sergey Plis and Vamsi Potluru, 2019. 

function [W,H,e,t] = sparseNMF(X,r,options)

timect = cputime; 
[m,n] = size(X); 
if nargin <= 2
    options = [];
end
if ~isfield(options,'sW')
    options.sW = []; 
end
if ~isfield(options,'sH')
    options.sH = []; 
end
if ~isfield(options,'maxiter')
    options.maxiter = 100; 
end
if ~isfield(options,'timemax')
    options.timemax = 5; 
end
if ~isfield(options,'delta')
    options.delta = 0.1; % Stop inner iter if improvment of current 
                         % iteration is not as good as delta * improvment 
                         % of the first one. 
end
if ~isfield(options,'inneriter')
    options.inneriter = 10; % Maximum number of inner iterations 
end
if ~isfield(options,'NeNMF')
    options.NeNMF = 0;   % Use of FGM for update of H 
end
if ~isfield(options,'colproj')
    options.colproj = 0;   % Use of column-wise projection for 
end
if isfield(options,'W') && isfield(options,'H') % Initialization 
    W = options.W;
    H = options.H; 
else
    W = rand(m,r); 
    H = rand(r,n); 
    % Optimal scaling  
    alpha = sum(sum( W'*X .* H) ) /  sum(sum( (W'*W).*(H*H') ) ); 
    W = alpha*W; 
end
% Projection of W to achieve average sparsity s 
if ~isempty(options.sW)
    W = weightedgroupedsparseproj_col(W,options.sW,options);
end
if ~isempty(options.sH)
    H = weightedgroupedsparseproj_col(H',options.sH,options);
    H = H'; 
end
itercount = 1; 
nX = norm(X,'fro')^2; 
e = []; 
t = []; 
Wbest = W; 
Hbest = H; 
ndisplay = 10; 
disp('Iteration number and error:') 
while itercount <= options.maxiter && cputime-timect <= options.timemax
    % Normalize W and H so that columns/rows have the same norm, that is, 
    % ||W(:,k)|| = ||H(k,:)|| for amll k 
    normW = sqrt((sum(W.^2)))+1e-16; 
    normH = sqrt((sum(H'.^2)))+1e-16; 
    Wo = W; Ho = H; 
    for k = 1 : r
        W(:,k) = W(:,k)/sqrt(normW(k))*sqrt(normH(k)); 
        H(k,:) = H(k,:)/sqrt(normH(k))*sqrt(normW(k)); 
    end
    % Update H: 
    % (1) No sparsity constraints: block coordinate descent method; 
    % See N. Gillis and F. Glineur, "Accelerated Multiplicative Updates 
    % and Hierarchical ALS Algorithms for Nonnegative Matrix Factorization", 
    % Neural Computation 24 (4), pp. 1085-1105, 2012.
    if options.NeNMF == 0 && isempty(options.sH)
        H = nnlsHALSupdt(X,W,H,options.delta,options.inneriter); 
    else
    % (2) With sparsity constraints: fast projected gradient method (FGM) 
    % similar to NeNMF from 
    % Guan, N., Tao, D., Luo, Z., & Yuan, B. NeNMF: An optimal 
    % gradient method for nonnegative matrix factorization, IEEE 
    % Transactions on Signal Processing, 60(6), 2882-2898, 2012.
        options.s = options.sH; 
        H = fastgradsparseNNLS(X',W',H',options); 
        H = H'; 
    end
    % Update W: same as for W
    if isempty(options.sW) % A-HALS
        [W,XHt,HHt] = nnlsHALSupdt(X',H',W',options.delta,options.inneriter); 
        W = W'; 
        XHt = XHt'; 
    else                   % NeNMF-like 
        options.s = options.sW; 
        [W,XHt,HHt] = fastgradsparseNNLS(X,H,W,options); 
    end
    % Time and error
    e = [e sqrt( max(0, (nX-2*sum(sum(W.*XHt))+ sum(sum(HHt.*(W'*W)))) ) )]; 
    t = [t cputime-timect]; 
    % Keep best iterate in memory as FGM is not guaranteed to be monotone
    if itercount >= 2 
        if e(end) <= e(end-1)
            Wbest = W; 
            Hbest = H; 
        end
    end
    % Display
    if mod(itercount,ndisplay) == 0
    	fprintf('%2.0f:%2.3f - ',itercount,e(itercount));
    end
    if mod(itercount,ndisplay*10) == 0
    	fprintf('\n');
    end
    itercount = itercount+1; 
end
if mod(itercount,ndisplay*10) > 0, fprintf('\n'); end 
W = Wbest; 
H = Hbest; 