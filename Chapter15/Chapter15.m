%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Computational illustration for Chapter 15
%       The Complexity of Sparse Recovery
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
% CVX is needed to run the second part of this file

%% generate a 2s-sparse vector and its first 2s Fourier observations
% select problem sizes
N = 500; 
s = 10;
m = 2*s;
% create the sparse vector x to be recovered
x = zeros(N,1);
aux = randperm(N);
supp_ori = sort(aux(1:s)); 
x(supp_ori) = randn(s,1);
% produce the observation vector y made of 2s Fourier coefficients 
xhat = fft(x); 
y_exact = xhat(1:m);
% as well as a noisy version
noise = 1e-5*rand(m,1); 
y_noisy = y_exact+noise;


%% Sparse recovery via Prony's method seems to be successful...
phat = zeros(N,1); 
phat(1) = 1;
M = toeplitz(y_exact(s:2*s-1),y_exact(s:-1:1));
phat(2:s+1) = -M\y_exact(s+1:2*s);
p = ifft(phat);
[~,idx] = sort(abs(p)); 
supp_exact = sort(idx(1:s))';
disp('In the exact case, the original and recovered supports agree:')
[supp_ori; supp_exact]


%% But it is not robust to observation errors with only m=2s

phat = zeros(N,1); 
phat(1) = 1;
M = toeplitz(y_noisy(s:2*s-1),y_noisy(s:-1:1));
phat(2:s+1) = -M\y_noisy(s+1:2*s);
p = ifft(phat);
[~,idx] = sort(abs(p)); 
supp_noisy = sort(idx(1:s))';
disp('In the noisy case, the original and recovered supports do not agree anymore:')
[supp_ori; supp_noisy]


%% The outputted vectors do not agree either

F = fft(eye(N));      % the full discrete Fourier matrix
A = F(1:2*s,:);       % the submatrix for the first 2s Fourier coefficients
x_exact = zeros(N,1);
x_exact(supp_exact) = A(:,supp_exact)\y_exact;
x_noisy = zeros(N,1);
x_noisy(supp_noisy) = A(:,supp_noisy)\y_noisy;
sprintf(strcat('Recovery from exact observations is quite successful: relative L2-error =',...
    32, num2str(norm(x-x_exact)/norm(x))))
sprintf(strcat('Recovery from inexact observations is not successful: relative L2-error =',...
    32, num2str(norm(x-x_noisy)/norm(x))))

%% In contrast, recovery via L1-minimization is stable
% (with more observations, of course)

m = 4*s;
A = F(1:m,:);
y_exact = A*x;
y_noisy = y_exact+1e-5*rand(m,1);
cvx_begin quiet
variable x1_exact(N)
minimize norm(x1_exact,1)
subject to
A*x1_exact == y_exact;
cvx_end
cvx_begin quiet
variable x1_noisy(N)
minimize norm(x1_noisy,1)
subject to
A*x1_noisy == y_noisy;
cvx_end
sprintf(strcat('Recovery from exact observations is quite successful: relative L2-error =',...
    32, num2str(norm(x-x1_exact)/norm(x))))
sprintf(strcat('Recovery from inexact observations is not bad either: relative L2-error =',...
    32, num2str(norm(x-x1_noisy)/norm(x))))
