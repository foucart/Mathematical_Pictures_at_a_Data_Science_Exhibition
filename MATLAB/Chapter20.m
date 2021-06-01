%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Computational illustration for Chapter 20
%        Snippets of Linear Programming     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
% CVX is needed to run this file 


%% Inscribed circle to the d-simplex

% first, the euclidean plane: the answer be 1/(2+sqrt(2))=0.2929
d = 2;                            % ambient dimension
p = 2;                            % index of the \ell_p-norm
q = p/(p-1);                      % dual index
A = [ones(1,d); -eye(d)];
b = [1; zeros(d,1)];
row_norms = zeros(d+1,1);
for i = 1:d+1
    row_norms(i) = norm(A(i,:),q);
end
cvx_begin quiet
variable center(d)
variable radius
maximize radius
subject to
A*center + radius*row_norms <= b
cvx_end
sprintf(strcat('In the euclidean plane, the inscribed radius is', ...
    32, num2str(radius)))

%% Guessing the value of the inscribed radius as a function of d (when p=2)
d_max = 10;
p = 2; q = 2;
radius = zeros(1,d_max);
for d=1:d_max
    A = [ones(1,d); -eye(d)];
    b = [1; zeros(d,1)];
    row_norms = zeros(d+1,1);
    for i = 1:d+1
        row_norms(i) = norm(A(i,:),q);
    end
    cvx_begin quiet
    variable c(d)
    variable r
    maximize r
    subject to
    A*c + r*row_norms <= b
    cvx_end
    radius(d) = r;
end
fprintf(strcat('For d form 1 to', 32, num2str(d_max), ...
    ', the values of the inscribed radii with p=', num2str(p), ' are'))
radius

%% Guessing the value of the inscribed radius as a function of d (when p=Inf)
d_max = 10;
p = inf; q = 1;
radius = zeros(1,d_max);
for d=1:d_max
    A = [ones(1,d); -eye(d)];
    b = [1; zeros(d,1)];
    row_norms = zeros(d+1,1);
    for i = 1:d+1
        row_norms(i) = norm(A(i,:),q);
    end
    cvx_begin quiet
    variable c(d)
    variable r
    maximize r
    subject to
    A*c + r*row_norms <= b
    cvx_end
    radius(d) = r;
end
fprintf(strcat('For d form 1 to', 32, num2str(d_max), ...
    ', the values of the inscribed radii with p=', num2str(p), ' are'))
radius

%% Guessing the value of the inscribed radius as a function of d (when p=1)
d_max = 10;
p = 1; q = inf;
radius = zeros(1,d_max);
for d=1:d_max
    A = [ones(1,d); -eye(d)];
    b = [1; zeros(d,1)];
    row_norms = zeros(d+1,1);
    for i = 1:d+1
        row_norms(i) = norm(A(i,:),q);
    end
    cvx_begin quiet
    variable c(d)
    variable r
    maximize r
    subject to
    A*c + r*row_norms <= b
    cvx_end
    radius(d) = r;
end
fprintf(strcat('For d form 1 to', 32, num2str(d_max), ...
    ', the values of the inscribed radii with p=', num2str(p), ' are'))
radius

%% Guessing the value of the inscribed radius as a function of d and p
% the answer seems to be: 1/[d^{1-1/p}*(d^{1/p}+1)] = 1/[d+d^{1/q}] 

%% Determination of the proximity (wrto the L_inf-norm)
% between the d-simplex and a rotated and shifted version of it

d = 16;
A = [ones(1,d); -eye(d)];
b = [1; zeros(d,1)];
Q = hadamard(d)/sqrt(d);
AA = A*Q';
bb = b + A*Q'*ones(d,1);
cvx_begin quiet
variable x(d)
variable xx(d)
variable c
minimize c
subject to
A*x <= b;
AA*xx <= bb;
x-xx <= c;
xx-x <= c; 
cvx_end
prox = c

%% Determination of the Hausdorff distance (wrto to the L_1-norm)
% between the d-simplex and a rotated and shifted version of it

d = 16;
A = [ones(1,d); -eye(d)];
b = [1; zeros(d,1)];
V = [zeros(d,1), eye(d)];    %columns of V = vertices of the d-simplex
Q = hadamard(d)/sqrt(d);
AA = A*Q';
bb = b + A*Q'*ones(d,1);
VV = Q*V + ones(d,d+1);      %columns of VV = vertices of other simplex
left_term = zeros(1,d+1);
right_term = zeros(1,d+1);
for k=1:d+1
   cvx_begin quiet
   variable xx(d)
   variable c(d)
   minimize sum(c)
   subject to
   AA*xx <= bb;
   V(:,k)-xx <= c;
   xx-V(:,k) <= c;
   cvx_end
   left_term(k) = cvx_optval;
   cvx_begin quiet
   variable x(d)
   variable c(d)
   minimize sum(c)
   subject to
   A*x <= b;
   VV(:,k)-x <= c;
   x-VV(:,k) <= c;
   cvx_end
   right_term(k) = cvx_optval;
end
Haus_dist = max(max(left_term),max(right_term))
