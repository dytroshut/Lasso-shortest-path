%% RGG: shortest path + ADMM + InADMM (weighted LASSO) with uniqueness via weight jitter & penalty tie-break
clear; clc; close all;

%% ---------------- User controls ----------------
n                = 3000;     % nodes in [0,1]^2
TARGET_GC_FRAC   = 0.98;
MAX_ITERS_RADIUS = 20;
R_INFLATE        = 1.18;

alpha     = 2;               % base weights: w0 = d^alpha
ADD_NOISE = true;
mu_rel    = 2e-2;            % multiplicative jitter: w <- w .* (1 + mu_rel*U[0,1])
nu_rel    = 1e-3;            % additive jitter:       w <- w + nu_rel*max(w)*U[0,1]
tie_rel   = 1e-12;           % deterministic weight tie-break (added after noise)

% ℓ1 penalty tie-break (weighted LASSO): λ_j = λ * (1 + penalty_tie_rel * ξ_j)
penalty_tie_rel = 1e-8;

% LASSO settings
lambda_scale = 1e-6;         % λ = lambda_scale * λ_max  (smaller => more edges)

% ADMM / InADMM options
opts.rho     = 1.0;
opts.alpha   = 1.8;
opts.maxit   = 5000;
opts.abstol  = 1e-8;
opts.reltol  = 1e-6;
opts.verbose = true;         % set false to silence

% InADMM (PCG) options
opts.cgtol   = 1e-8;
opts.maxcg   = 2000;

% Plot styling
NODE_SIZE_SMALL  = 1.2;
BASE_EDGE_ALPHA  = 0.55;
BASE_EDGE_WIDTH  = 0.9;
PATH_EDGE_WIDTH  = 4.0;
CLR_SP = [0 0 0];            % black
CLR_AD = [0.85 0 0];         % dark red
CLR_IN = [0 0.45 0];         % dark green

% Display threshold for ADMM/InADMM edges (no post shortestpath)
thr_abs = 1e-5;
rng(42);

%% ---------------- Sample node positions ----------------
X = rand(n,1); Y = rand(n,1);

%% ---------------- Radius: grow until giant comp meets target ---------------
r = sqrt( max(log(n),1) / (pi*n) ); 
r = min(max(r, 1e-4), sqrt(2));
r = 1.1*r;

G = [];
for it = 1:MAX_ITERS_RADIUS
    [I,J] = rgg_edges_radius(X, Y, r);
    Gtmp  = graph(I, J, [], n);
    comp  = conncomp(Gtmp);
    sz    = accumarray(comp',1,[max(comp),1]);
    gc    = max(sz)/n;
    fprintf('  r-iter %2d: r=%.4g, edges=%d, giant frac=%.2f%%\n', it, r, numedges(Gtmp), 100*gc);
    if gc >= TARGET_GC_FRAC, G = Gtmp; break; else, r = min(R_INFLATE*r, sqrt(2)); end
end
if isempty(G), warning('Using last graph (target GC not reached).'); G = Gtmp; end

% Keep giant component
comp = conncomp(G);
sz   = accumarray(comp',1,[max(comp),1]);
[~, gid] = max(sz);
keep = (comp == gid);
G = subgraph(G, keep);
Xg = X(keep); Yg = Y(keep);

ng = numnodes(G); m = numedges(G);
fprintf('Kept giant component: n=%d, m=%d (final r=%.4g)\n', ng, m, r);

%% ---------------- Weights: base + multiplicative/additive jitter + tie-break
EN   = G.Edges.EndNodes; sIdx = EN(:,1); tIdx = EN(:,2);
d    = hypot(Xg(sIdx)-Xg(tIdx), Yg(sIdx)-Yg(tIdx));
w0   = max(d, eps).^alpha;

if ADD_NOISE
    M     = max(w0);
    xi1   = rand(m,1);                   % multiplicative U[0,1]
    xi2   = rand(m,1);                   % additive      U[0,1]
    w_n   = w0 .* (1 + mu_rel*xi1) + (nu_rel*M)*xi2;
else
    w_n   = w0;
end
w_n(w_n<=0) = eps;

% deterministic tie-break in weights (to kill residual ties)
w = add_tiebreak(w_n, sIdx, tIdx, tie_rel);
G.Edges.Weight = w;

%% ---------------- s (left-most) & t (farthest reachable) -------------------
rx = (Xg - min(Xg))/max(1e-12, max(Xg)-min(Xg));
[~, start_node] = min(rx);
distAll = distances(G, start_node);         % uses current w
cands   = find(isfinite(distAll)); cands(cands==start_node) = [];
[~, jj] = max(distAll(cands));
end_node = cands(jj);

%% ---------------- Built-in shortest path (baseline) -----------------------
[pathSP, ~] = shortestpath(G, start_node, end_node, 'Method','auto');
eSP = findedge(G, pathSP(1:end-1), pathSP(2:end));

%% ---------------- Build Q = D W^{-1}, y = e_s - e_t -----------------------
I_D = [sIdx;  tIdx]; J_D = [(1:m)'; (1:m)']; V_D = [ones(m,1); -ones(m,1)];
D   = sparse(I_D, J_D, V_D, ng, m);
Q   = D * spdiags(1./w, 0, m, m);

y = sparse(ng,1); y(start_node)=1; y(end_node)=-1;

%% ---------------- λ and per-edge λ_j (penalty tie-break) -------------------
lambda_max = norm(Q'*y, inf);
lambda     = lambda_scale * lambda_max;
% tiny per-edge variations in the ℓ1 penalty (forces a unique minimizer)
tau = 1 + penalty_tie_rel * edge_hash01(sIdx, tIdx);   % in (1, 1+ε)
lambda_vec = lambda * tau;

fprintf('lambda_max=%.3e, lambda=%.3e, mu_rel=%.1e, nu_rel=%.1e, tie_rel=%.1e, penalty_tie_rel=%.1e\n',...
        lambda_max, lambda, mu_rel, nu_rel, tie_rel, penalty_tie_rel);

%% ---------------- ADMM (weighted LASSO) -----------------------------------
[z0,v0,a0] = deal(zeros(m,1));
% [zAD, aAD, vAD, histAD] = admm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0);
% eAD = find(abs(zAD) >= thr_abs);
% fprintf('ADMM: iters=%d, r=%.2e, s=%.2e, obj=%.4e, selected edges=%d/%d\n', ...
%         histAD.iters, histAD.rnorm(end), histAD.snorm(end), histAD.obj(end), numel(eAD), m);

%% ---------------- InADMM (weighted LASSO with PCG) ------------------------
[zIN, aIN, vIN, histIN] = inadmm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0);

thr_abs = 1e-3;
eIN = find(abs(zIN) >= thr_abs);
fprintf('InADMM: iters=%d, r=%.2e, s=%.2e, obj=%.4e, selected edges=%d/%d\n', ...
        histIN.iters, histIN.rnorm(end), histIN.snorm(end), histIN.obj(end), numel(eIN), m);

%% ---------------- Plots -------------------------
% 1) Built-in SP (black)
figure('Color','w'); 
p1 = base_plot(G, Xg, Yg, NODE_SIZE_SMALL, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
if ~isempty(eSP), highlight(p1,'Edges',eSP,'LineWidth',PATH_EDGE_WIDTH,'EdgeColor',CLR_SP); end
mark_st(Xg, Yg, start_node, end_node);
%title(sprintf('RGG — built-in shortest path (n=%d, m=%d)', ng, m));

%%
% 2) ADMM: edges with |z| >= thr_abs (dark red)
% figure('Color','w'); 
% p2 = base_plot(G, Xg, Yg, NODE_SIZE_SMALL, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
% if ~isempty(eAD), highlight(p2,'Edges',eAD,'LineWidth',PATH_EDGE_WIDTH,'EdgeColor',CLR_AD); end
% mark_st(Xg, Yg, start_node, end_node);
%title(sprintf('RGG — ADMM edges with |z| \\ge %.1e', thr_abs));

%%
% 3) InADMM: edges with |z| >= thr_abs (dark green)
figure('Color','w'); 
p3 = base_plot(G, Xg, Yg, NODE_SIZE_SMALL, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
if ~isempty(eIN), highlight(p3,'Edges',eIN,'LineWidth',PATH_EDGE_WIDTH,'EdgeColor',CLR_IN); end
mark_st(Xg, Yg, start_node, end_node);
title(sprintf('RGG — InADMM edges with |z| \\ge %.1e', thr_abs));

%% ---------------- Convergence figures ----------
% plot_history(histAD, 'ADMM (weighted)');
plot_history(histIN, 'InADMM (weighted)');

%% =================== Helpers ===================

function [I,J] = rgg_edges_radius(X,Y,r)
% Return undirected edges (i<j) for all pairs within radius r.
    n = numel(X); I = []; J = [];
    if exist('createns','file')==2 && exist('rangesearch','file')==2
        ns  = createns([X Y],'NSMethod','kdtree');
        idx = rangesearch(ns, [X Y], r);
        for i=1:n
            nbrs = idx{i};
            nbrs = nbrs(nbrs > i);
            if ~isempty(nbrs)
                I = [I; i*ones(numel(nbrs),1)];
                J = [J; nbrs(:)];
            end
        end
    elseif exist('pdist','file')==2 && exist('squareform','file')==2
        D = squareform(pdist([X Y]));
        [ii,jj] = find(triu(D <= r, 1));
        I = ii; J = jj;
    else
        blk = 1000;
        for i1=1:blk:n
            i2 = min(n, i1+blk-1);
            Xi = X(i1:i2); Yi = Y(i1:i2);
            for j1=i1:blk:n
                j2 = min(n, j1+blk-1);
                Xj = X(j1:j2)'; Yj = Y(j1:j2)';
                Dx = Xi - Xj; Dy = Yi - Yj;
                D2 = hypot(Dx, Dy) <= r;
                if i1==j1, D2 = triu(D2,1); end
                [ii,jj] = find(D2);
                I = [I; (i1-1)+ii];
                J = [J; (j1-1)+jj];
            end
        end
    end
end

function wU = add_tiebreak(w, sIdx, tIdx, rel)
    w = double(w(:));
    M = max(w(~isinf(w))); if isempty(M), M = 1; end
    eps0 = rel * M;
    phi  = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
    tb   = abs((phi*43758.5453) - floor(phi*43758.5453)); % [0,1)
    wU   = w + eps0*tb;
end

function h = edge_hash01(sIdx,tIdx)
% Deterministic pseudo-random in [0,1) per edge
    phi = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
    h   = abs((phi*43758.5453) - floor(phi*43758.5453));
end

function p = base_plot(G,X,Y,nodeSize,edgeAlpha,edgeWidth)
    p = plot(G, ...
        'XData', X, 'YData', Y, ...
        'NodeLabel', {}, ...
        'NodeColor', 'k', ...
        'Marker', '.', ...
        'MarkerSize', nodeSize, ...
        'EdgeColor', [0 0.4470 0.7410], ...
        'EdgeAlpha', edgeAlpha, ...
        'LineWidth', edgeWidth);
    axis equal off; hold on;
end

function mark_st(X,Y,s,t)
    plot(X([s t]),Y([s t]),'ko','MarkerFaceColor','w','MarkerSize',6);
    text(X(s),Y(s),'  s','FontWeight','bold','Color','k');
    text(X(t),Y(t),'  t','FontWeight','bold','Color','k');
end

function plot_history(H, name)
    figure('Color','w');
    subplot(2,1,1);
    semilogy(1:numel(H.rnorm), H.rnorm,'-','LineWidth',1.4); hold on;
    semilogy(1:numel(H.snorm), H.snorm,'--','LineWidth',1.4);
    grid on; legend('primal r','dual s','Location','southwest');
    xlabel('iteration'); ylabel('residual'); title([name ' residuals']);
    subplot(2,1,2);
    semilogy(1:numel(H.obj), H.obj,'-','LineWidth',1.4); grid on;
    xlabel('iteration'); ylabel('objective'); title([name ' objective']);
end

%% ======== ADMM: weighted LASSO (z,a,v) ========
function [z, a, v, hist] = admm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0)
    rho = opts.rho; alpha = getf(opts,'alpha',1.8);
    maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
    [~,m] = size(Q); z=z0; a=a0; v=v0; lamv=lambda_vec(:);
    A = (Q'*Q) + rho*speye(m); rhs0 = Q'*y;
    useChol=true; try R = chol(A,'lower'); catch, useChol=false; end
    hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
    for k=1:maxit
        rhs = rhs0 + rho*(a - v);
        if useChol, z = R'\(R\rhs); else, z = A\rhs; end
        z_hat = alpha*z + (1-alpha)*a;
        a_old = a;
        a = soft_vec(z_hat + v, lamv/rho);   % elementwise λ_j/ρ
        v = v + (z_hat - a);
        r = z - a; s = rho*(a - a_old);
        rnorm=norm(r); snorm=norm(s);
        eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
        eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
        hist.rnorm(k)=rnorm; hist.snorm(k)=snorm;
        hist.obj(k)=0.5*norm(Q*z - y)^2 + sum(lamv.*abs(z));
        hist.iters=k;
        if verbose && (mod(k,50)==0 || k==1)
            fprintf('ADMM %4d  r=%.2e  s=%.2e  obj=%.4e\n',k,rnorm,snorm,hist.obj(k));
        end
        if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
    end
    hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
end



%% ======== InADMM (PCG z-update via (QQ'+ρI)^{-1} identity) ================
% function [z, a, v, hist] = inadmm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0)
%     rho   = opts.rho; alpha=getf(opts,'alpha',1.8);
%     maxit = opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
%     cgtol = getf(opts,'cgtol',1e-8); maxcg=getf(opts,'maxcg',2000);
%     [~,m] = size(Q); z=z0; a=a0; v=v0; lamv=lambda_vec(:);
%     rhs0 = Q'*y;
%     % preconditioner diag(QQ' + rho I)
%     dPc = full(sum(Q.^2,2) + rho); 
%     Mfun = @(x) x ./ dPc;             % diagonal preconditioner
%     Aop  = @(x) Q*(Q'*x) + rho*x;     % (QQ' + rho I)x
%     hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
%     for k=1:maxit
%         h   = rhs0 + rho*(a - v);
%         b   = Q*h;
%         [eta, flag] = pcg(Aop, b, cgtol, maxcg, Mfun, [], []);
%         if verbose && flag~=0
%             fprintf('  PCG flag=%d (k=%d)\n',flag,k);
%         end
%         z = (1/rho)*(h - Q'*eta);
%         z_hat= alpha*z + (1-alpha)*a;
%         a_old= a;
%         a = soft_vec(z_hat + v, lamv/rho);
%         v = v + (z_hat - a);
%         r = z - a; s = rho*(a - a_old);
%         rnorm=norm(r); snorm=norm(s);
%         eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
%         eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
%         hist.rnorm(k)=rnorm; hist.snorm(k)=snorm;
%         hist.obj(k)=0.5*norm(Q*z - y)^2 + sum(lamv.*abs(z));
%         hist.iters=k;
%         if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
%     end
%     hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
% end
% 
% %% -------- utilities --------
% function z = soft_vec(x,tau), z = sign(x).*max(abs(x)-tau,0); end
% function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end


%%
function [z, a, v, hist] = inadmm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0, sIdx, tIdx, w)
% InADMM with PCG z-update, using ichol preconditioner, adaptive tol, warm-starts.
% Extra inputs (needed once to build preconditioner): sIdx,tIdx (edge endpoints), w (edge weights)

    rho   = opts.rho; alpha=getf(opts,'alpha',1.8);
    maxit = opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
    baseTol = getf(opts,'cgtol',1e-4);  % looser base tol
    maxcg   = getf(opts,'maxcg',2000);

    [n,m] = size(Q); z=z0; a=a0; v=v0; lamv=lambda_vec(:);
    rhs0 = Q'*y;

    % ---------- Build L_rho = Q Q' + rho I as a sparse Laplacian + rho I ----------
    % Note: Q = D * W^{-1}  => Q Q' = D * W^{-2} * D'
    w2  = 1./(w(:).^2);
    Ii  = sIdx(:); Jj = tIdx(:);
    L   = sparse(n,n);
    L = L + sparse(Ii,Ii,w2,n,n) + sparse(Jj,Jj,w2,n,n) ...
          - sparse(Ii,Jj,w2,n,n) - sparse(Jj,Ii,w2,n,n);
    L = L + rho*speye(n);

    % ---------- ichol preconditioner (fallback to diagonal if it fails) ----------
    useICHOL = true;
    try
        setup.type    = 'ict';
        setup.droptol = 1e-3;
        setup.michol  = 'off';
        setup.diagcomp= 0.1;      % add a bit on the diagonal to stabilize
        R = ichol(L, setup);      % L ≈ R*R'
        M1 = R; M2 = R';
    catch
        if verbose
            warning('ichol failed; falling back to diagonal preconditioner.');
        end
        useICHOL = false;
        dPc = full(diag(L));
        M1 = @(x) x ./ dPc;
        M2 = [];
    end

    % PCG operator: A x = (QQ' + rho I) x
    Aop  = @(x) Q*(Q'*x) + rho*x;

    % History
    hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;

    % Warm-start for eta
    eta = zeros(n,1);

    for k=1:maxit
        % Adaptive CG tol: loose early, tighter later
        cgtol_k = max(baseTol, 1e-2/sqrt(k));  % e.g., 1e-2, 7e-3, ..., floor at baseTol

        h   = rhs0 + rho*(a - v);
        b   = Q*h;

        % Warm-start PCG using previous eta
        if useICHOL
            [eta, flag] = pcg(Aop, b, cgtol_k, maxcg, M1, M2, eta);
        else
            [eta, flag] = pcg(Aop, b, cgtol_k, maxcg, M1, M2, eta);
        end
        if verbose && flag~=0
            fprintf('  PCG k=%d flag=%d (tol=%.1e)\n',k,flag,cgtol_k);
        end

        % z update via identity
        z = (1/rho)*(h - Q'*eta);

        % a,v updates
        z_hat= alpha*z + (1-alpha)*a;
        a_old= a;
        a = soft_vec(z_hat + v, lamv/rho);
        v = v + (z_hat - a);

        % residuals & stopping
        r = z - a; s = rho*(a - a_old);
        rnorm=norm(r); snorm=norm(s);
        eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
        eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
        hist.rnorm(k)=rnorm; hist.snorm(k)=snorm;
        hist.obj(k)=0.5*norm(Q*z - y)^2 + sum(lamv.*abs(z));
        hist.iters=k;

        if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
    end

    hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
end

function z = soft_vec(x,tau), z = sign(x).*max(abs(x)-tau,0); end
function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end