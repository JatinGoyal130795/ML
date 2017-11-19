function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
[k,n]=size(X);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h=zeros(size(y));
for i=1:m
kj=0;
kj = kj+(X(i,:)*theta);
h(i,1)=sigmoid(kj);
endfor
% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
for i=1:m
kj = (1/m)*(-y(i,1)* log(h(i,1)) - (1 - y(i,1))* log(1-h(i,1)));
J=J+kj;
endfor
t=h.-y;
for j=1:n
kj = (1/m)*X(:,j)*t;
grad(j)=grad(j)+sum(kj);
endfor






% =============================================================

end
