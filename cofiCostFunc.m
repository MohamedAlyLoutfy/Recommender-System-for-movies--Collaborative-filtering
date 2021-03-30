function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

  
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));





[r c]=find(R);
l=size(r);
predictions=zeros(l,1);
for z=1:l
  i=r(z);
  j=c(z);
  XT=X(i,:);
  prediction=Theta(j,:)*XT';
  real=Y(i,j);
  error=(prediction-real).^2;
  J=J+error;
endfor
J=1/2*J;
J=J+ lambda / 2 * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));

for i=1:num_movies
    idx = find(R(i, :) == 1);
    tempTheta = Theta(idx, :);
    tempY = Y(i, idx);
    X_grad(i, :) = (X(i, :) * tempTheta' - tempY) * tempTheta + lambda * X(i, :) ;
end

for i=1:num_users
    idx = find(R(:, i) == 1);
    tempX = X(idx, :);
    tempY = Y(idx, i);
    Theta_grad(i, :) = (tempX * Theta(i, :)' - tempY)' * tempX + lambda * Theta(i, :) ;
end




grad = [X_grad(:); Theta_grad(:)];

end
