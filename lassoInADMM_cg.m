function [z, history] = lassoInADMM_cg(A, b, lambda, rho, alpha)
% lasso  Solve lasso problem via ADMM
%
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*|| Ax - b ||_2^2 + \lambda || x ||_1
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).

% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

t_start = tic;

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-6;  %-6
RELTOL   = 1e-6;

[p, n] = size(A);

A_sp = sparse(A);
% save a matrix-vector multiply
H_hat =  A_sp*A_sp'./rho + speye(p);

Atb = A'*b;

% The tolerance sigma
%sigma = 1/(1+norm(A)./sqrt(2*rho)); % fixed sigma
sigma = 1e-8;   % -8 nice %-2 fastest

eta = zeros(p,1);
z   = zeros(n,1);
u   = zeros(n,1);
x = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % Define h(k) and e_k(eta) 
    h = Atb + rho*(z - u); %h(k)
    % x-update 
    y = A*h./rho;  %y A h rho
    % Conjugate gradient to solve the system of lieanr equations 
    eta = conjgrad(H_hat, y, eta, sigma); %H_hat y eta sigma
    x = h./rho-A'*eta./rho;      % x = h rho A eta 

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, lambda/rho);

    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x, z);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    history.znorm(k) = norm(z,1);
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end

if ~QUIET
    toc(t_start);
end
end

function x = conjgrad(A, b, x, sigma)
    % A = sparse(A);
    r = b - A * x;
    p = r;
    rsold = r' * r;
    tolerance = sqrt(rsold)*sigma;
    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < tolerance
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

function p = objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) );
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end
