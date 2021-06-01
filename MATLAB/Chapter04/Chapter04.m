%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Computational Illustration for Chapter 04
%            Support Vector Machines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;

%% The case of two linearly separable classes
% blue dots labeled negatively, red crosses labeled positively 

%% generate the data 
m0 = 100;
m1 = 50;
X0 = zeros(m0,2); 
X0(:,1) = rand(m0,1);                          % abscissae of the blue dots
X0(:,2) = 3/2*X0(:,1) + 1/2 + 0.2*randn(m0,1); % their ordinates
X1 = zeros(m1,2);
X1(:,1) = 1 + rand(m1,1);                      % abscissae of the red crosses
X1(:,2) = rand(m1,1);                          % their ordinates
figure(1)
plot(X0(:,1),X0(:,2),'bo',X1(:,1),X1(:,2),'r+')
hold on

%% separating "hyperplane" produced by a linear feasibility problem
cvx_quiet true
%cvx_solver gurobi            % uncomment if gurobi is to be used
cvx_begin
variable w_feas(2)
variable b_feas
minimize 1
subject to
X0*w_feas-b_feas <= -1;
X1*w_feas-b_feas >= +1;
cvx_end
% visualize the "hyperplane" (in green)
grid_x = 0:0.1:2;
plot(grid_x,(-w_feas(1)*grid_x+b_feas)/w_feas(2),'g');

%% separating "hyperplane" produced by the perceptron algorithm
X = [X0 ones(m0,1); X1 ones(m1,1)];
y = [-ones(m0,1); +ones(m1,1)];
w_perc = zeros(3,1);
obj = y.*(X*w_perc);     % to be entrywise positive at the end on the loop 
while min(obj) <= 0;
    [~,i] = min(obj);
    w_perc = w_perc + (y(i)/norm(X(i,:))^2)*X(i,:)';
    obj = y.*(X*w_perc);
end
% visualize the "hyperplane" (in black)
plot(grid_x,(-w_perc(1)*grid_x-w_perc(3))/w_perc(2),'k');

%% separating "hyperplane" produced by hard SVM
cvx_begin
variable w_hard(2)
variable b_hard
minimize norm(w_hard)
subject to 
X0*w_hard-b_hard <= -1;
X1*w_hard-b_hard >= +1;
cvx_end
% visualize the "hyperplane" (in magenta)
plot(grid_x,(-w_hard(1)*grid_x+b_hard)/w_hard(2),'m');


%% The case of two almost linearly separable classes
% blue dots labeled negatively, red crosses labeled positively

%% generate the data 
clear all; clc;
m0 = 100;
m1 = 50;
X0 = zeros(m0,2); 
X0(:,1) = rand(m0,1);                          % abscissae of the blue dots
X0(:,2) = X0(:,1) + 1/4 + 0.2*randn(m0,1);     % their ordinates
X1 = zeros(m1,2);
X1(:,1) = 0.6 + rand(m1,1);                    % abscissae of the red crosses
X1(:,2) = rand(m1,1);                          % their ordinates
figure(2)
plot(X0(:,1),X0(:,2),'bo',X1(:,1),X1(:,2),'r+')
hold on

%% "hyperplane" produced by soft SVM
cvx_quiet true
%cvx_solver gurobi            % uncomment if gurobi is to be used
lambda = 1e-3;
cvx_begin
variable w_soft(2)
variable b_soft
variable xi_soft(m0+m1) nonnegative
minimize sum(w_soft.*w_soft) + (1/lambda)*sum(xi_soft)
subject to 
X0*w_soft-b_soft <= -1 + xi_soft(1:m0);
X1*w_soft-b_soft >= +1 - xi_soft(m0+1:end);
cvx_end
% visualize the "hyperplane" (in black)
grid_x = 0:0.1:2;
plot(grid_x,(-w_soft(1)*grid_x+b_soft)/w_soft(2),'k');


%% First case of two classes clearly not linearly separable
% blue dots labeled negatively, red crosses labeled positively

%% generate the data 
clear all; clc;
m0 = 100;
m1 = 50;
X0 = zeros(m0,2); 
X0(:,1) = 6*rand(m0,1);                          % abscissae of the blue dots
X0(:,2) = sin(X0(:,1)) + 0.5*rand(m0,1);         % their ordinates
X1 = zeros(m1,2);
X1(:,1) = 6*rand(m1,1);                          % abscissae of the red crosses
X1(:,2) = sin(X1(:,1)) - 0.5*rand(m1,1);         % their ordinates
figure(3)
plot(X0(:,1),X0(:,2),'bo',X1(:,1),X1(:,2),'r+')
hold on

%% Soft SVM with a polynomial kernel
X = [X0; X1];
y = [-ones(m0,1); +ones(m1,1)];
K_poly = (1 + X*X').^3;
cvx_quiet true
lambda = 1e-3;
cvx_begin
variable a_poly(m0+m1)
variable b_poly
variable xi_poly(m0+m1) nonnegative
minimize ( a_poly'*K_poly*a_poly + (1/lambda)*sum(xi_poly) )
subject to
y.*(K_poly*a_poly-b_poly) >= 1-xi_poly
cvx_end
% visualize the separating surface
f = @(u,v) a_poly'*(1+X*[u;v]).^3 - b_poly;
ezplot(f,[0,6,-2,2])
title('')


%% Second case of two classes clearly not linearly separable
% blue dots labeled negatively, red crosses labeled positively

%% generate the data
clear all; clc;
m0 = 50;
m1 = 100;
r0 = 1.1*sqrt(rand(m0,1));
theta0 = 2*pi*rand(m0,1);
r1 = 0.8+sqrt(rand(m1,1));
theta1 = 2*pi*rand(m1,1);
X = [r0.*cos(theta0), r0.*sin(theta0);...
     r1.*cos(theta1), r1.*sin(theta1)];
y = [-ones(m0,1); +ones(m1,1)];
figure(4)
plot(X(1:m0,1),X(1:m0,2),'bo',X(m0+1:end,1),X(m0+1:end,2),'r+')
hold on

%% Soft SVM with a gaussian kernel

sigma = 1;
m = m0+m1;
K_gauss = zeros(m,m);
for i = 1:m
    for j = 1:m
    K_gauss(i,j) = exp(-norm(X(i,:)-X(j,:))^2/2/sigma^2);
    end
end
cvx_quiet true
lambda = 2e-1;
cvx_begin
variable a_gauss(m)
variable b_gauss
variable xi_gauss(m) nonnegative
minimize ( a_gauss'*K_gauss*a_gauss + (1/lambda)*sum(xi_gauss) )
subject to
y.*(K_gauss*a_gauss-b_gauss) >= 1-xi_gauss
cvx_end
% visualize the separating surface
f = @(u,v) a_gauss'*exp( -(diag(X*X') - 2*X*[u;v] + u^2+v^2)/2/sigma^2 )...
    - b_gauss; 
ezplot(f,[-2,2,-2,2])
title('')
