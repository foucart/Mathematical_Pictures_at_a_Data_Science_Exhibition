%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Computational illustration for Chapter 16
%  Low-Rank Recovery from Linear Observations   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
% CVX is needed to run this file
% comment out the next line if MOSEK is not installed
cvx_solver mosek  

%% create a rank-r matrix to be recovered
n = 60;
r = 3;
X = randn(n,r)*rand(r,n);
m = 1000;           % number of observations made on X

%% verify that X is succesfully recovered by nuclear norm minimization
% with a generic observation scheme
y_gen = zeros(m,1);
A = randn(n,n,m);
for i=1:m
    y_gen(i) = trace(A(:,:,i)'*X);
end
cvx_begin quiet
variable X_gen(n,n)
variable P(n,n)
variable Q(n,n)
minimize trace(P)+trace(Q)
subject to
for i=1:m
    trace(A(:,:,i)'*X_gen) == y_gen(i);
end
[P X_gen; X_gen' Q] == semidefinite(2*n);
cvx_end
sprintf(strcat('Recovery considered to be exact, with a relative Frobenius-error of'...
    , 32, num2str(norm(X-X_gen,'fro')/norm(X,'fro'))))

%% verify that X is succesfully recovered by nuclear norm minimization
% with a rank-one observation scheme
y_rk1 = zeros(m,1);
a = randn(n,m);
b = randn(n,m);
for i=1:m
    y_rk1(i) = b(:,i)'*X*a(:,i);
end
cvx_begin quiet
variable X_rk1(n,n)
variable P(n,n)
variable Q(n,n)
minimize trace(P)+trace(Q)
subject to
for i=1:m
    b(:,i)'*X_rk1*a(:,i) == y_rk1(i);
end
[P X_rk1; X_rk1' Q] == semidefinite(2*n);
cvx_end
sprintf(strcat('Recovery considered to be exact, with a relative Frobenius-error of'...
    , 32, num2str(norm(X-X_rk1,'fro')/norm(X,'fro'))))

%% verify that X is succesfully recovered by nuclear norm minimization
% with an entrywise observation scheme
Omega = randperm(n*n,m);
y_ent = X(Omega);
cvx_begin quiet
variable X_ent(n,n)
variable P(n,n)
variable Q(n,n)
minimize trace(P)+trace(Q)
subject to
X_ent(Omega) == y_ent;
[P X_ent; X_ent' Q] == semidefinite(2*n);
cvx_end
sprintf(strcat('Recovery (often) considered to be exact, with a relative Frobenius-error of'...
    , 32, num2str(norm(X-X_ent,'fro')/norm(X,'fro'))))
