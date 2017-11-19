function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m ,n]=size(X);
% number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=zeros(size(y));
for i=1:m
kj =(X(i,:)*theta);
h(i,1)=sigmoid(kj);
endfor
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
for i=1:m
kj = (1/m)*(-y(i,1)* log(h(i,1)) - (1 - y(i,1))* log(1-h(i,1)));
J=J+kj;
endfor

kp=theta.^2;
J=J+((lambda/(2*m))*sum(kp));

for i=1:m
kj = (1/m)*X(i,1)*((h(i,1) - y(i,1)));
grad(1)=grad(1)+kj;
endfor

t=h.-y;
for j=2:n
ks = (1/m)*(X(:,j).*t);
grad(j)=grad(j)+sum(ks);
km=(lambda/m)*theta(j,1);
grad(j)=grad(j)+km;
endfor





% =============================================================

end
