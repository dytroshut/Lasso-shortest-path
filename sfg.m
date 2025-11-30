%% BA(n=5000,m=3) with arbitrary random edge weights in [1,100]
% Dijkstra baseline (bright red) + InADMM (fast direct chol) path highlighting
clear; clc; close all;

%% ---------------- User controls ----------------
n  = 5000;     % nodes
m  = 3;        % edges per new node
seedBA = 42;   % RNG for BA generation

% Arbitrary random weights (unique a.s.)
w_min = 1; 
w_max = 100;          % draw weights ~ U([1,100])
tie_rel = 1e-12;      % tiny deterministic tiebreak relative to max(w)

% LASSO / InADMM
lambda_scale      = 1e-5;   % lambda = lambda_scale * lambda_max
penalty_tie_rel   = 1e-8;   % per-edge λ_j = λ*(1+ε_j) for unique minimizer
opts.rho     = 1.0;
opts.alpha   = 1.8;
opts.maxit   = 10000;
opts.abstol  = 1e-8;
opts.reltol  = 1e-6;
opts.verbose = true;

% Display
thr_abs          = 2;        % show |z| >= thr_abs for InADMM
NODE_MARKER      = '.';         % tiny dots so path pops
NODE_SIZE_SMALL  = 1.0;         % very small nodes
BASE_EDGE_COLOR  = [0.30 0.35 0.45];
BASE_EDGE_ALPHA  = 0.08;        % faint background edges
BASE_EDGE_WIDTH  = 0.5;
PATH_EDGE_WIDTH  = 4.5;

CLR_RED          = [1.00 0.00 0.00];  % Bright red for BOTH plots
HALO_ON          = true;              % white halo for extra pop
HALO_COLOR       = [1 1 1];
HALO_EXTRA_W     = 2.5;

rng(42); % reproducible

%% ---------------- BA graph (simple, undirected, unique edges) --------------
G = ba_scale_free(n, m, [], seedBA);   % numeric nodes 1..n
ng = numnodes(G); me = numedges(G);
fprintf('BA graph: n=%d, m=%d, |E|=%d\n', ng, m, me);

EN   = G.Edges.EndNodes;
sIdx = EN(:,1); tIdx = EN(:,2);

%% ---------------- Arbitrary random positive weights in [1,100] -------------
w_rand = w_min + (w_max - w_min) * rand(me,1);   % U([1,100])
w      = add_tiebreak(w_rand, sIdx, tIdx, tie_rel);  % tiny deterministic tiebreak
G.Edges.Weight = w;

%% ---------------- Pick far-apart s,t under these weights -------------------
% Double sweep for weighted eccentricity endpoints
s0 = randi(ng);
d1 = distances(G, s0); [~, s] = max(d1);
d2 = distances(G, s);  [~, t] = max(d2);
fprintf('Endpoints: s=%d, t=%d (approx diameter endpoints under random w)\n', s, t);

%% ---------------- Dijkstra shortest path (baseline) ------------------------
[pathSP, ~] = shortestpath(G, s, t, 'Method','auto');
eSP = findedge(G, pathSP(1:end-1), pathSP(2:end));

%% ---------------- LASSO (implicit Q): D, y, lambda, lambda_vec -------------
I = [sIdx;  tIdx]; J = [(1:me)'; (1:me)']; V = [ones(me,1); -ones(me,1)];
D = sparse(I, J, V, ng, me);

y = sparse(ng,1); y(s)=1; y(t)=-1;

% lambda_max = ||Q'^y||_inf = ||(D' y)./w||_inf (no explicit Q form)
rhs0        = (D' * y) ./ w;
lambda_max  = norm(rhs0, inf);
lambda      = lambda_scale * lambda_max;

% tiny per-edge λ_j tie-break (unique minimizer)
tau         = 1 + penalty_tie_rel * edge_hash01(sIdx, tIdx);
lambda_vec  = lambda * tau;

fprintf('lambda_max=%.3e, lambda=%.3e, weights~U([%g,%g]), tie_rel=%.1e, penalty_tie_rel=%.1e\n',...
        lambda_max, lambda, w_min, w_max, tie_rel, penalty_tie_rel);

%% ---------------- InADMM (fast direct chol; no explicit Q) -----------------
[z0, v0, a0] = deal(zeros(me,1));
[zIN, aIN, vIN, histIN] = inadmm_lasso_weighted_direct_implicit(D, sIdx, tIdx, w, y, lambda_vec, opts);

eIN = find(abs(zIN) >= thr_abs);
fprintf('InADMM: iters=%d, r=%.2e, s=%.2e, obj=%.4e, selected edges=%d/%d\n', ...
        histIN.iters, histIN.rnorm(end), histIN.snorm(end), histIN.obj(end), numel(eIN), me);

%% ---------------- One force layout (reused for overlays) -------------------
f0 = figure('Color','w');
p0 = plot(G, 'Layout','force', 'Iterations', 100, ...
    'NodeLabel', {}, ...
    'Marker', NODE_MARKER, 'MarkerSize', NODE_SIZE_SMALL, ...
    'EdgeColor', BASE_EDGE_COLOR, 'EdgeAlpha', BASE_EDGE_ALPHA, 'LineWidth', BASE_EDGE_WIDTH);
axis off; title('BA (n=5000,m=3) — faint background'); drawnow;
Xc = p0.XData(:); Yc = p0.YData(:); % freeze coords

%% ---------------- Plot: Dijkstra path (bright red + white halo) ------------
f1 = figure('Color','w');
p1 = base_plot_with_coords(G, Xc, Yc, NODE_MARKER, NODE_SIZE_SMALL, BASE_EDGE_COLOR, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
if ~isempty(eSP)
    if HALO_ON
        highlight(p1, 'Edges', eSP, 'LineWidth', PATH_EDGE_WIDTH + HALO_EXTRA_W, 'EdgeColor', HALO_COLOR);
    end
    highlight(p1, 'Edges', eSP, 'LineWidth', PATH_EDGE_WIDTH, 'EdgeColor', CLR_RED);
end
mark_st(Xc, Yc, s, t);
title('Shortest path (Dijkstra)');

%% ---------------- Plot: InADMM support (same bright red + halo) ------------
f2 = figure('Color','w');
p2 = base_plot_with_coords(G, Xc, Yc, NODE_MARKER, NODE_SIZE_SMALL, BASE_EDGE_COLOR, BASE_EDGE_ALPHA, BASE_EDGE_WIDTH);
if ~isempty(eIN)
    if HALO_ON
        highlight(p2, 'Edges', eIN, 'LineWidth', PATH_EDGE_WIDTH + HALO_EXTRA_W, 'EdgeColor', HALO_COLOR);
    end
    highlight(p2, 'Edges', eIN, 'LineWidth', PATH_EDGE_WIDTH, 'EdgeColor', CLR_RED);
end
mark_st(Xc, Yc, s, t);
title(sprintf('InADMM edges with |z| \\ge %.1e', thr_abs));

%% ---------------- Convergence (same format as before) ----------------------
plot_history(histIN, 'InADMM (direct)');

%% ======================================================================
%% ====================== LOCAL FUNCTIONS BELOW =========================
%% ======================================================================

function G = ba_scale_free(n, m, m0, seed)
%BA_SCALE_FREE  Barabási–Albert graph (undirected, simple, unique edges).
    if nargin < 2 || isempty(m),  m  = 3; end
    if nargin < 3 || isempty(m0), m0 = max(m,5); end
    if nargin >= 4 && ~isempty(seed), rng(seed); end
    assert(n >= m0 && m0 >= m && m >= 1, 'Require n>=m0>=m>=1.');

    % Preallocate and fill initial clique
    M_est = m0*(m0-1)/2 + m*(n - m0);
    s = zeros(M_est,1);  t = zeros(M_est,1);  eidx = 0;
    for i=1:m0-1
        for j=i+1:m0
            eidx = eidx+1; s(eidx)=i; t(eidx)=j;
        end
    end

    % Urn with multiplicities = degrees
    deg = zeros(n,1); deg(1:m0) = m0-1;
    urn = repelem(1:m0, deg(1:m0));

    % Preferential attachment
    for new = (m0+1):n
        targets = [];
        while numel(targets) < m
            pick_idx = randperm(numel(urn), min(m - numel(targets), numel(urn)));
            cand = unique(urn(pick_idx), 'stable');
            cand = setdiff(cand, targets, 'stable');
            targets = [targets, cand]; %#ok<AGROW>
        end
        for k=1:m
            eidx = eidx+1; s(eidx)=new; t(eidx)=targets(k);
        end
        deg(targets) = deg(targets) + 1;
        deg(new)     = deg(new) + m;
        urn = [urn, targets, repmat(new, 1, m)]; %#ok<AGROW>
    end

    % Unique undirected edges, no self-loops
    s = s(1:eidx); t = t(1:eidx);
    a = min(s,t); b = max(s,t);
    E = unique([a b], 'rows');
    E = E(E(:,1)~=E(:,2), :);

    G = graph(E(:,1), E(:,2));
end

function wU = add_tiebreak(w, sIdx, tIdx, rel)
% Tiny deterministic tie-break scaled to max(w)
    w = double(w(:));
    M = max(w(~isinf(w))); if isempty(M), M = 1; end
    eps0 = rel * M;
    phi  = sin(double(sIdx).*12.9898 + double(tIdx).*78.233);
    tb   = abs((phi*43758.5453) - floor(phi*43758.5453));  % in [0,1)
    wU   = w + eps0*tb;
end

function h = edge_hash01(sIdx,tIdx)
% Deterministic pseudo-random in [0,1) per edge
    phi = sin(double(sIdx).*12.9898 + double(tIdx).*78.233);
    h   = abs((phi*43758.5453) - floor(phi*43758.5453));
end

function p = base_plot_with_coords(G,X,Y,marker,ms,edgeColor,edgeAlpha,edgeWidth)
% Plot using fixed coordinates so overlays align perfectly
    p = plot(G, ...
        'XData', X, 'YData', Y, ...
        'NodeLabel', {}, ...
        'Marker', marker, 'MarkerSize', ms, ...
        'EdgeColor', edgeColor, 'EdgeAlpha', edgeAlpha, 'LineWidth', edgeWidth);
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

%% ======== InADMM (direct chol; no explicit Q) ==============================
function [z, a, v, hist] = inadmm_lasso_weighted_direct_implicit(D, sIdx, tIdx, w, y, lambda_vec, opts)
% InADMM with exact inner solve via sparse Cholesky of L = D*W^{-2}*D' + rho*I.
% Uses identities (no explicit Q):
%   Q' y = (D' y) ./ w
%   Q h  = D * (h ./ w)
%   z    = (1/rho) * ( h - (D' eta) ./ w )

    rho     = opts.rho;    alpha   = getf(opts,'alpha',1.8);
    maxit   = opts.maxit;  abstol  = opts.abstol;  reltol = opts.reltol;
    verbose = getf(opts,'verbose',false);

    [n,m] = size(D); z=zeros(m,1); a=z; v=z; lamv=lambda_vec(:);
    rhs0  = (D' * y) ./ w;

    % Build L = D*W^{-2}*D' + rho I as a weighted Laplacian + diagonal
    w2  = 1./(w(:).^2);
    L   = sparse(n,n);
    L = L + sparse(sIdx,sIdx,w2,n,n) + sparse(tIdx,tIdx,w2,n,n) ...
          - sparse(sIdx,tIdx,w2,n,n) - sparse(tIdx,sIdx,w2,n,n);
    L = L + rho*speye(n);

    % AMD + sparse Cholesky (once)
    p   = symamd(L); Lp = L(p,p); dL = decomposition(Lp,'chol');

    hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;

    for k=1:maxit
        h     = rhs0 + rho*(a - v);     % m×1
        b     = D * (h ./ w);           % n×1  (Qh)
        bp    = b(p);
        etap  = dL \ bp;                % solve (QQ'+rho I) eta = Q h
        eta   = zeros(n,1); eta(p) = etap;

        z = (1/rho) * (h - (D' * eta) ./ w);

        z_hat = alpha*z + (1-alpha)*a;
        a_old = a;
        a     = soft_vec(z_hat + v, lamv/rho);
        v     = v + (z_hat - a);

        % Residuals & objective (no explicit Q)
        r = z - a; s = rho*(a - a_old);
        rnorm=norm(r); snorm=norm(s);
        eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
        eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
        qz    = D * (z ./ w) - y;       % Qz - y
        hist.rnorm(k)=rnorm; hist.snorm(k)=snorm; hist.obj(k)=0.5*(qz.'*qz) + sum(lamv.*abs(z));
        hist.iters=k;

        if verbose && (mod(k,50)==0 || k==1)
            fprintf('InADMM %4d  r=%.2e  s=%.2e  obj=%.4e\n',k,rnorm,snorm,hist.obj(k));
        end
        if (rnorm<=eps_p) && (snorm<=eps_d), break; end
    end
    hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
end

function z = soft_vec(x,tau), z = sign(x).*max(abs(x)-tau,0); end
function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end