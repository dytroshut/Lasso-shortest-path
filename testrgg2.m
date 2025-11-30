%% RGG + LASSO (ADMM only) with uniqueness via weight jitter + penalty tie-break
% Fancy dark plotting (saturated colors, halo)
clear; clc; close all;

%% ---------------- User controls ----------------
n                = 6000;     % nodes in [0,1]^2
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

% ADMM settings
lambda_scale = 1e-7;         % λ = lambda_scale * λ_max  (smaller => more edges)
opts.rho     = 1.0;
opts.alpha   = 1.8;
opts.maxit   = 5000;
opts.abstol  = 1e-8;
opts.reltol  = 1e-6;
opts.verbose = true;

% -------- Plot styling (dark theme) --------
BG                = [0.05 0.06 0.08];  % background
BASE_EDGE_COL     = [0.80 0.88 1.00];  % faint bluish edges on dark bg
BASE_EDGE_ALPHA   = 0.35;
BASE_EDGE_WIDTH   = 0.9;
NODE_DOT_COL      = [0.90 0.92 0.98];
NODE_DOT_SIZE     = 1.2;

CLR_AD            = [1.00 0.10 0.10];  % vivid red (ADMM)
CLR_SP            = [1.00 0.80 0.00];  % strong yellow (built-in shortest path)
HALO_ON           = true;
HALO_COL          = [1 1 1]*0.92;
PATH_EDGE_WIDTH   = 4.2;
HALO_EXTRA        = 2.0;               % halo width added to path width

SHOW_SHORTESTPATH = false;             % set true to overlay built-in SP

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

% deterministic tie-break in weights (keeps path uniqueness under equalities)
w = add_tiebreak(w_n, sIdx, tIdx, tie_rel);
G.Edges.Weight = w;

%% ---------------- s (left-most) & t (farthest reachable) -------------------
rx = (Xg - min(Xg))/max(1e-12, max(Xg)-min(Xg));
[~, start_node] = min(rx);
distAll = distances(G, start_node);         % weighted by w
cands   = find(isfinite(distAll)); cands(cands==start_node) = [];
[~, jj] = max(distAll(cands));
end_node = cands(jj);

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
[zAD, aAD, vAD, histAD] = admm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0);

%% ---------------- Threshold & selection -----------------------------------
thr_abs = 3e-5;
eAD = find(abs(zAD) >= thr_abs);
fprintf('ADMM: iters=%d, r=%.2e, s=%.2e, obj=%.4e, selected edges=%d/%d\n', ...
        histAD.iters, histAD.rnorm(end), histAD.snorm(end), histAD.obj(end), numel(eAD), m);

%% ---------------- Fancy dark plot (RGG) -----------------------------------
% Optional shortest path for reference
if SHOW_SHORTESTPATH
    [pathSP, ~] = shortestpath(G, start_node, end_node, 'Method','auto');
    eSP = findedge(G, pathSP(1:end-1), pathSP(2:end));
else
    eSP = [];
end

figure('Color', BG);
ax = axes('Color', BG); hold(ax, 'on');
p = plot(G, ...
    'XData', Xg, 'YData', Yg, ...
    'NodeLabel', {}, ...
    'NodeColor', NODE_DOT_COL, ...
    'Marker', '.', ...
    'MarkerSize', NODE_DOT_SIZE, ...
    'EdgeColor', BASE_EDGE_COL, ...
    'EdgeAlpha', BASE_EDGE_ALPHA, ...
    'LineWidth', BASE_EDGE_WIDTH, ...
    'Parent', ax);
axis(ax, 'equal'); axis(ax, 'off');

% Overlay ADMM path (with halo)
if ~isempty(eAD)
    [XE, YE] = build_edge_poly(Xg, Yg, G.Edges.EndNodes, eAD);
    if HALO_ON
        plot(ax, XE, YE, '-', 'LineWidth', PATH_EDGE_WIDTH+HALO_EXTRA, 'Color', HALO_COL);
    end
    plot(ax, XE, YE, '-', 'LineWidth', PATH_EDGE_WIDTH, 'Color', [0.00 1.00 0.35]);
end

% Optional: built-in shortest path in yellow
if ~isempty(eSP)
    [XS, YS] = build_edge_poly(Xg, Yg, G.Edges.EndNodes, eSP);
    if HALO_ON
        plot(ax, XS, YS, '-', 'LineWidth', PATH_EDGE_WIDTH+HALO_EXTRA, 'Color', HALO_COL);
    end
    plot(ax, XS, YS, '-', 'LineWidth', PATH_EDGE_WIDTH, 'Color', CLR_AD);
end

% [1.00 0.10 0.10];  % vivid red (ADMM)
% CLR_SP            = [1.00 0.80 0.00];


% s / t markers & labels (white on dark)
scatter(ax, Xg([start_node end_node]), Yg([start_node end_node]), 40, ...
    'o', 'filled', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 1.0);
text(ax, Xg(start_node), Yg(start_node), '  s', 'Color', [1 1 1], ...
    'FontWeight', 'bold', 'FontSize', 14, 'VerticalAlignment', 'middle');
text(ax, Xg(end_node),   Yg(end_node),   '  t', 'Color', [1 1 1], ...
    'FontWeight', 'bold', 'FontSize', 14, 'VerticalAlignment', 'middle');

%title(ax, sprintf('RGG — ADMM edges with |z| \\ge %.1e (n=%d, m=%d)', thr_abs, ng, m), ...
    %'Color', [0.95 0.97 1.00]);

%% ---------------- Convergence -------------------
plot_history(histAD, 'ADMM (weighted)');

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
        % Blocked brute force (no toolboxes)
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
% Deterministic tiny perturbation to ensure unique path choices
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

function [XE, YE] = build_edge_poly(X, Y, EndNodes, eList)
% Build NaN-separated polylines for a set of edges (indices in eList).
    if isempty(eList)
        XE = []; YE = []; return;
    end
    s = EndNodes(eList,1);
    t = EndNodes(eList,2);
    k = numel(eList);
    XE = [X(s)'; X(t)'; nan(1,k)]; XE = XE(:);
    YE = [Y(s)'; Y(t)'; nan(1,k)]; YE = YE(:);
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

%% ======== ADMM (weighted LASSO) ========
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
        % elementwise soft-threshold with per-edge λ_j/ρ
        a = soft_vec(z_hat + v, lamv/rho);
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

function z = soft_vec(x,tau)
% elementwise soft-threshold: tau may be a vector
    z = sign(x) .* max(abs(x)-tau, 0);
end

function val = getf(s,f,def)
    if isfield(s,f), val=s.(f); else, val=def; end
end





% %% RGG + LASSO (ADMM only) with uniqueness via weight jitter + penalty tie-break
% clear; clc; close all;
% 
% %% ---------------- User controls ----------------
% n                = 3000;     % nodes in [0,1]^2
% TARGET_GC_FRAC   = 0.98;
% MAX_ITERS_RADIUS = 20;
% R_INFLATE        = 1.18;
% 
% alpha     = 2;               % base weights: w0 = d^alpha
% ADD_NOISE = true;
% mu_rel    = 2e-2;            % multiplicative jitter: w <- w .* (1 + mu_rel*U[0,1])
% nu_rel    = 1e-3;            % additive jitter:       w <- w + nu_rel*max(w)*U[0,1]
% tie_rel   = 1e-12;           % deterministic weight tie-break (added after noise)
% 
% % ℓ1 penalty tie-break (weighted LASSO): λ_j = λ * (1 + penalty_tie_rel * ξ_j)
% penalty_tie_rel = 1e-8;
% 
% % ADMM settings
% lambda_scale = 1e-6; %1e-6        % λ = lambda_scale * λ_max  (smaller => more edges)
% opts.rho     = 1.0;
% opts.alpha   = 1.8;
% opts.maxit   = 5000;
% opts.abstol  = 1e-8;
% opts.reltol  = 1e-6;
% opts.verbose = true;
% 
% % Plot styling
% NODE_SIZE_SMALL  = 1.2;
% BASE_EDGE_ALPHA  = 0.55;
% BASE_EDGE_WIDTH  = 0.9;
% PATH_EDGE_WIDTH  = 4.0;
% CLR_AD = [0.85 0 0];
% 
% rng(42);
% 
% %% ---------------- Sample node positions ----------------
% X = rand(n,1); Y = rand(n,1);
% 
% %% ---------------- Radius: grow until giant comp meets target ---------------
% r = sqrt( max(log(n),1) / (pi*n) ); 
% r = min(max(r, 1e-4), sqrt(2));
% r = 1.1*r;
% 
% G = [];
% for it = 1:MAX_ITERS_RADIUS
%     [I,J] = rgg_edges_radius(X, Y, r);
%     Gtmp  = graph(I, J, [], n);
%     comp  = conncomp(Gtmp);
%     sz    = accumarray(comp',1,[max(comp),1]);
%     gc    = max(sz)/n;
%     fprintf('  r-iter %2d: r=%.4g, edges=%d, giant frac=%.2f%%\n', it, r, numedges(Gtmp), 100*gc);
%     if gc >= TARGET_GC_FRAC, G = Gtmp; break; else, r = min(R_INFLATE*r, sqrt(2)); end
% end
% if isempty(G), warning('Using last graph (target GC not reached).'); G = Gtmp; end
% 
% % Keep giant component
% comp = conncomp(G);
% sz   = accumarray(comp',1,[max(comp),1]);
% [~, gid] = max(sz);
% keep = (comp == gid);
% G = subgraph(G, keep);
% Xg = X(keep); Yg = Y(keep);
% 
% ng = numnodes(G); m = numedges(G);
% fprintf('Kept giant component: n=%d, m=%d (final r=%.4g)\n', ng, m, r);
% 
% %% ---------------- Weights: base + multiplicative/additive jitter + tie-break
% EN   = G.Edges.EndNodes; sIdx = EN(:,1); tIdx = EN(:,2);
% d    = hypot(Xg(sIdx)-Xg(tIdx), Yg(sIdx)-Yg(tIdx));
% w0   = max(d, eps).^alpha;
% 
% if ADD_NOISE
%     M     = max(w0);
%     xi1   = rand(m,1);                   % multiplicative U[0,1]
%     xi2   = rand(m,1);                   % additive      U[0,1]
%     w_n   = w0 .* (1 + mu_rel*xi1) + (nu_rel*M)*xi2;
% else
%     w_n   = w0;
% end
% w_n(w_n<=0) = eps;
% 
% % deterministic tie-break in weights (keeps path uniqueness under equalities)
% w = add_tiebreak(w_n, sIdx, tIdx, tie_rel);
% G.Edges.Weight = w;
% 
% %% ---------------- s (left-most) & t (farthest reachable) -------------------
% rx = (Xg - min(Xg))/max(1e-12, max(Xg)-min(Xg));
% [~, start_node] = min(rx);
% distAll = distances(G, start_node);         % uses current w
% cands   = find(isfinite(distAll)); cands(cands==start_node) = [];
% [~, jj] = max(distAll(cands));
% end_node = cands(jj);
% 
% %% ---------------- Build Q = D W^{-1}, y = e_s - e_t -----------------------
% I_D = [sIdx;  tIdx]; J_D = [(1:m)'; (1:m)']; V_D = [ones(m,1); -ones(m,1)];
% D   = sparse(I_D, J_D, V_D, ng, m);
% Q   = D * spdiags(1./w, 0, m, m);
% 
% y = sparse(ng,1); y(start_node)=1; y(end_node)=-1;
% 
% %% ---------------- λ and per-edge λ_j (penalty tie-break) -------------------
% lambda_max = norm(Q'*y, inf);
% lambda     = lambda_scale * lambda_max;
% % tiny per-edge variations in the ℓ1 penalty (forces a unique minimizer)
% tau = 1 + penalty_tie_rel * edge_hash01(sIdx, tIdx);   % in (1, 1+ε)
% lambda_vec = lambda * tau;
% 
% fprintf('lambda_max=%.3e, lambda=%.3e, mu_rel=%.1e, nu_rel=%.1e, tie_rel=%.1e, penalty_tie_rel=%.1e\n',...
%         lambda_max, lambda, mu_rel, nu_rel, tie_rel, penalty_tie_rel);
% 
% %% ---------------- ADMM (weighted LASSO) -----------------------------------
% [z0,v0,a0] = deal(zeros(m,1));
% [zAD, aAD, vAD, histAD] = admm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0);
% 
% %%
% thr_abs = 3e-5;
% eAD = find(abs(zAD) >= thr_abs);
% fprintf('ADMM: iters=%d, r=%.2e, s=%.2e, obj=%.4e, selected edges=%d/%d\n', ...
%         histAD.iters, histAD.rnorm(end), histAD.snorm(end), histAD.obj(end), numel(eAD), m);
% 
% % ---------------- Plot: base + ADMM-selected edges -------------------------
% figure('Color','w'); 
% p = base_plot(G, Xg, Yg, NODE_SIZE_SMALL, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
% if ~isempty(eAD)
%     highlight(p, 'Edges', eAD, 'LineWidth', PATH_EDGE_WIDTH, 'EdgeColor', CLR_AD);
% end
% mark_st(Xg, Yg, start_node, end_node);
% title(sprintf('RGG — ADMM edges with |z| \\ge %.1e (n=%d, m=%d)', thr_abs, ng, m));
% 
% %% ---------------- Convergence -------------------
% plot_history(histAD, 'ADMM (weighted)');
% 
% %% =================== Helpers ===================
% 
% function [I,J] = rgg_edges_radius(X,Y,r)
% % Return undirected edges (i<j) for all pairs within radius r.
%     n = numel(X); I = []; J = [];
%     if exist('createns','file')==2 && exist('rangesearch','file')==2
%         ns  = createns([X Y],'NSMethod','kdtree');
%         idx = rangesearch(ns, [X Y], r);
%         for i=1:n
%             nbrs = idx{i};
%             nbrs = nbrs(nbrs > i);
%             if ~isempty(nbrs)
%                 I = [I; i*ones(numel(nbrs),1)];
%                 J = [J; nbrs(:)];
%             end
%         end
%     elseif exist('pdist','file')==2 && exist('squareform','file')==2
%         D = squareform(pdist([X Y]));
%         [ii,jj] = find(triu(D <= r, 1));
%         I = ii; J = jj;
%     else
%         blk = 1000;
%         for i1=1:blk:n
%             i2 = min(n, i1+blk-1);
%             Xi = X(i1:i2); Yi = Y(i1:i2);
%             for j1=i1:blk:n
%                 j2 = min(n, j1+blk-1);
%                 Xj = X(j1:j2)'; Yj = Y(j1:j2)';
%                 Dx = Xi - Xj; Dy = Yi - Yj;
%                 D2 = hypot(Dx, Dy) <= r;
%                 if i1==j1, D2 = triu(D2,1); end
%                 [ii,jj] = find(D2);
%                 I = [I; (i1-1)+ii];
%                 J = [J; (j1-1)+jj];
%             end
%         end
%     end
% end
% 
% function wU = add_tiebreak(w, sIdx, tIdx, rel)
%     w = double(w(:));
%     M = max(w(~isinf(w))); if isempty(M), M = 1; end
%     eps0 = rel * M;
%     phi  = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
%     tb   = abs((phi*43758.5453) - floor(phi*43758.5453)); % [0,1)
%     wU   = w + eps0*tb;
% end
% 
% function h = edge_hash01(sIdx,tIdx)
% % Deterministic pseudo-random in [0,1) per edge
%     phi = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
%     h   = abs((phi*43758.5453) - floor(phi*43758.5453));
% end
% 
% function p = base_plot(G,X,Y,nodeSize,edgeAlpha,edgeWidth)
%     p = plot(G, ...
%         'XData', X, 'YData', Y, ...
%         'NodeLabel', {}, ...
%         'NodeColor', 'k', ...
%         'Marker', '.', ...
%         'MarkerSize', nodeSize, ...
%         'EdgeColor', [0 0.4470 0.7410], ...
%         'EdgeAlpha', edgeAlpha, ...
%         'LineWidth', edgeWidth);
%     axis equal off; hold on;
% end
% 
% function mark_st(X,Y,s,t)
%     plot(X([s t]),Y([s t]),'ko','MarkerFaceColor','w','MarkerSize',6);
%     text(X(s),Y(s),'  s','FontWeight','bold','Color','k');
%     text(X(t),Y(t),'  t','FontWeight','bold','Color','k');
% end
% 
% function plot_history(H, name)
%     figure('Color','w');
%     subplot(2,1,1);
%     semilogy(1:numel(H.rnorm), H.rnorm,'-','LineWidth',1.4); hold on;
%     semilogy(1:numel(H.snorm), H.snorm,'--','LineWidth',1.4);
%     grid on; legend('primal r','dual s','Location','southwest');
%     xlabel('iteration'); ylabel('residual'); title([name ' residuals']);
%     subplot(2,1,2);
%     semilogy(1:numel(H.obj), H.obj,'-','LineWidth',1.4); grid on;
%     xlabel('iteration'); ylabel('objective'); title([name ' objective']);
% end
% 
% %% ======== ADMM (weighted LASSO) ========
% function [z, a, v, hist] = admm_lasso_weighted(Q, y, lambda_vec, opts, z0, a0, v0)
%     rho = opts.rho; alpha = getf(opts,'alpha',1.8);
%     maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
%     [~,m] = size(Q); z=z0; a=a0; v=v0; lamv=lambda_vec(:);
%     A = (Q'*Q) + rho*speye(m); rhs0 = Q'*y;
%     useChol=true; try R = chol(A,'lower'); catch, useChol=false; end
%     hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
%     for k=1:maxit
%         rhs = rhs0 + rho*(a - v);
%         if useChol, z = R'\(R\rhs); else, z = A\rhs; end
%         z_hat = alpha*z + (1-alpha)*a;
%         a_old = a;
%         % elementwise soft-threshold with per-edge λ_j/ρ
%         a = soft_vec(z_hat + v, lamv/rho);
%         v = v + (z_hat - a);
%         r = z - a; s = rho*(a - a_old);
%         rnorm=norm(r); snorm=norm(s);
%         eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
%         eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
%         hist.rnorm(k)=rnorm; hist.snorm(k)=snorm;
%         hist.obj(k)=0.5*norm(Q*z - y)^2 + sum(lamv.*abs(z));
%         hist.iters=k;
%         if verbose && (mod(k,50)==0 || k==1)
%             fprintf('ADMM %4d  r=%.2e  s=%.2e  obj=%.4e\n',k,rnorm,snorm,hist.obj(k));
%         end
%         if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
%     end
%     hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
% end
% 
% function z = soft_vec(x,tau)
% % elementwise soft-threshold: tau may be a vector
%     z = sign(x) .* max(abs(x)-tau, 0);
% end
% 
% function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end