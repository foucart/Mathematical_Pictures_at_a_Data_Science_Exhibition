%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Computational illustration for Chapter 21
%          Duality Theory and Practice        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
% CVX is needed to run this file 

%% Circumscribed circle to the d-simplex wrto the L_inf-norm

% first, the euclidean plane: the answer be 1/2
d = 2;                            % ambient dimension
p = 2;                            % index of the \ell_p-norm
q = p/(p-1);                      % dual index
A = [ones(1,d); -eye(d)];
b = [1; zeros(d,1)];
cvx_begin quiet
variable center(d)
variable radius
variable Yp(d+1,d)
variable Ym(d+1,d)
minimize radius
subject to 
Yp >= 0;
Ym >= 0;
A'*Yp == +eye(d);
A'*Ym == -eye(d);
Yp'*b <= radius + center;
Ym'*b <= radius - center;
cvx_end
radius

%% Owl-norm minimization for sparse recovery
% warning: select a solver than run the simplex algorithm
cvx_solver mosek      

% Consider an observation matrix whose last two columns are identical...
N = 200;
m = 100;
A_aux = randn(m,N-1);
A = [A_aux A_aux(:,N-1)];
% ...and a sparse vector with last two entries being equal
x = zeros(N,1);
s = 10;
supp_aux = sort(randperm(N-2,s-1));
x(supp_aux) = randn(s-1,1);
x(N) = 1/2; x(N-1) = 1/2;
% produce the observation vector 
y = A*x;                             
% attempt to recover x from y by L1 norm minimization
cvx_begin quiet
variable xL1(N);
variable c(N);
minimize sum(c)
subject to 
A*xL1 == y;
c + xL1 >= 0;
c - xL1 >= 0;
cvx_end
% attemp to recover x from y by OWL norm minimization
w = sort(rand(N,1),'descend');
cvx_begin quiet
variable xOWL(N);
variable a(N);
variable b(N);
minimize sum(a)+sum(b)
subject to
A*xOWL == y;
repmat(a,1,N) + repmat(b',N,1) >= +w*xOWL';
repmat(a,1,N) + repmat(b',N,1) >= -w*xOWL';
cvx_end
%
sprintf(strcat('Recovery by L1 norm minimization is unsuccesful:', 32,...
    'the L1 error is', 32, num2str(norm(x-xL1,1)),10,...
    'Recovery by OWL norm minimization is succesful:', 32, 32,...
    'the L1 error is', 32, num2str(norm(x-xOWL,1))))

%% Verification of duality in semidefinite programming

d = 6;
n = 10;
primal_value = inf;
while abs(primal_value) == inf
C = randn(d,d); C = C+C';
A = zeros(d,d,n);
for i=1:n
    aux = randn(d,d);
    A(:,:,i) = aux+aux'; 
end
b = rand(n,1);
% the primal problem
cvx_begin quiet
variable X(d,d) semidefinite
minimize trace(C'*X)
subject to
for i=1:n
   trace(A(:,:,i)'*X) == b(i); 
end
cvx_end
primal_value = cvx_optval;
% the dual problem
cvx_begin quiet
variable nu(n)
expression M
M = zeros(d,d);
for i=1:n
    M = M + nu(i)*A(:,:,i);
end
maximize -sum(b.*nu)
subject to
M + C == semidefinite(d)
cvx_end
dual_value = cvx_optval;
end
%
sprintf(strcat('The optimal values of the two problems agree:',...
    32, 'primal value =', 32, num2str(primal_value),...
    32, ', dual value =', 32, num2str(dual_value)))