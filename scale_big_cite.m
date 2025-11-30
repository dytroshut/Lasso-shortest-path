%% Amsterdam (5k & 10k) — InADMM weighted LASSO (fast, direct chol) + dark basemap
% Plots ONLY the green InADMM support; no yellow overlay anywhere.
clear; clc; close all;

%% ---------------- Batch files ----------------
fileList = { ...
    'ams_5k_graph.mat', ...
    'ams_10k_graph.mat' ...
};

%% ---------------- Solver & plotting controls ----------------
% Weight jitter (ensures uniqueness / robustness)
ADD_NOISE       = true;
mu_rel          = 2e-2;        % multiplicative: w <- w .* (1 + mu_rel*U[0,1])
nu_rel          = 1e-3;        % additive:      w <- w + nu_rel*max(w)*U[0,1]
tie_rel         = 1e-12;       % tiny deterministic tiebreak on weights
penalty_tie_rel = 1e-8;        % tiny per-edge variation in l1 penalty

% LASSO lambda
lambda_scale    = 1e-7;        % lambda = lambda_scale * lambda_max

% InADMM options (fast: direct solve via sparse Cholesky)
opts.rho     = 1.0;
opts.alpha   = 1.8;
opts.maxit   = 10000;
opts.abstol  = 1e-8;
opts.reltol  = 1e-6;
opts.verbose = true;

% Display threshold for edges (no subgraph shortestpath)
thr_abs = 1e-4;

% Fancy plotting (dark map or planar)
USE_BASEMAP    = true;
BASEMAP_NAME   = 'streets-dark';

% Colors & sizes (dark-friendly, saturated)
BASE_EDGE_COL   = [0.90 0.94 1.00];   % bright bluish-gray base network
BASE_EDGE_W     = 1.1;
BASE_EDGE_ALPHA = 0.45;

INADMM_COL_GRN  = [0.00 1.00 0.35];   % vivid green for InADMM

HALO_ON       = true;
HALO_COL      = [1 1 1]*0.92;
HALO_W_EXTRA  = 2.0;
HILITE_LW     = 4.2;

NODE_DOT_SZ   = 7;   % planar nodes

rng(42);

%% ---------------- Run each file ----------------
for f = 1:numel(fileList)
    fprintf('\n=============== %s ===============\n', fileList{f});
    % -------- Load graph --------
    S   = load(fileList{f});
    fns = fieldnames(S);
    G   = S.(fns{1});                          % graph or digraph
    if isa(G,'digraph')
        G = graph(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), [], numnodes(G));
    end

    % -------- Node coordinates detection --------
    tryNames = {'X','Y'; 'x','y'; 'Lon','Lat'; 'lon','lat'; 'Longitude','Latitude'};
    Xc=[]; Yc=[];
    for k=1:size(tryNames,1)
        if ismember(tryNames{k,1}, G.Nodes.Properties.VariableNames) && ...
           ismember(tryNames{k,2}, G.Nodes.Properties.VariableNames)
            Xc = G.Nodes.(tryNames{k,1}); Yc = G.Nodes.(tryNames{k,2}); break;
        end
    end
    if isempty(Xc) || isempty(Yc), error('Node coordinates not found.'); end
    if iscell(Xc), Xc = cell2mat(Xc); end
    if iscell(Yc), Yc = cell2mat(Yc); end
    Xc = double(Xc(:)); Yc = double(Yc(:));

    % Geo detection (Lon/Lat) with heuristic fallback
    isGeo = false;
    nv = G.Nodes.Properties.VariableNames;
    if any(strcmpi(nv,'Lon')) || any(strcmpi(nv,'Longitude')), isGeo = true; end
    if ~isGeo
        if max(abs(Xc))<=180 && max(abs(Yc))<=90 && range(Xc)<10 && range(Yc)<10
            isGeo = true;
        end
    end

    % -------- Edges & base weights --------
    EN = G.Edges.EndNodes;
    if iscell(EN) || isstring(EN) || iscategorical(EN)
        sIdx = findnode(G, EN(:,1));  tIdx = findnode(G, EN(:,2));
    else
        sIdx = EN(:,1);               tIdx = EN(:,2);
    end
    m = numedges(G); n = numnodes(G);

    if ismember('Weight', G.Edges.Properties.VariableNames)
        w0 = double(G.Edges.Weight);
    else
        % Euclidean in given coordinates (planar or degrees — OK for routing weights)
        w0 = hypot(Xc(sIdx) - Xc(tIdx), Yc(sIdx) - Yc(tIdx));
    end
    w0 = double(w0(:)); w0(~isfinite(w0)) = inf; w0(w0<=0) = eps;

    % -------- Weight jitter + tiebreak --------
    if ADD_NOISE
        M   = max(w0(~isinf(w0))); if isempty(M), M = 1; end
        xi1 = rand(m,1); xi2 = rand(m,1);
        w_n = w0 .* (1 + mu_rel*xi1) + (nu_rel*M)*xi2;
    else
        w_n = w0;
    end
    w_n(w_n<=0) = eps;
    w = add_tiebreak(w_n, sIdx, tIdx, tie_rel);
    G.Edges.Weight = w;

    % -------- Pick s (left-most) and t (robust far-right) --------
    [ start_node, end_node, diagInfo ] = pick_left_to_right_nodes(G, Xc, Yc);
    fprintf('s=%d (left)  t=%d (right)  tried=%d  Δrx=%.3f  cands=%d\n',...
        start_node, end_node, diagInfo.numStartTried, diagInfo.finalDeltaRX, diagInfo.numCandsFinal);

    % -------- Build D (incidence), y, and lambda --------
    I_D = [sIdx;  tIdx]; J_D = [(1:m)'; (1:m)']; V_D = [ones(m,1); -ones(m,1)];
    D   = sparse(I_D, J_D, V_D, n, m);
    y   = sparse(n,1); y(start_node)=1; y(end_node)=-1;

    % lambda_max = ||Q'^y||_inf = ||(D' y)./w||_inf  (no explicit Q)
    rhs0        = (D' * y) ./ w;
    lambda_max  = norm(rhs0, inf);
    lambda      = lambda_scale * lambda_max;
    tau         = 1 + penalty_tie_rel * edge_hash01(sIdx, tIdx);
    lambda_vec  = lambda * tau;

    fprintf('lambda_max=%.3e, lambda=%.3e  (mu=%.1e, nu=%.1e, tie=%.1e, penalty_tie=%.1e)\n',...
        lambda_max, lambda, mu_rel, nu_rel, tie_rel, penalty_tie_rel);

    % -------- InADMM (direct chol; no explicit Q) --------
    [zIN, aIN, vIN, histIN] = inadmm_lasso_weighted_direct_implicit(D, sIdx, tIdx, w, y, lambda_vec, opts);

    % -------- Select edges --------
    eIN = find(abs(zIN) >= thr_abs);
    fprintf('InADMM: iters=%d  r=%.2e  s=%.2e  obj=%.4e  selected=%d/%d\n', ...
        histIN.iters, histIN.rnorm(end), histIN.snorm(end), histIN.obj(end), numel(eIN), m);

    % -------- Plot (geo basemap if Lon/Lat; else planar) --------
    if USE_BASEMAP && isGeo
        ax = base_plot_map(G, Xc, Yc, sIdx, tIdx, BASEMAP_NAME, BASE_EDGE_W, BASE_EDGE_COL, BASE_EDGE_ALPHA);
        if ~isempty(eIN)
            highlight_edges_map(ax, Xc, Yc, sIdx, tIdx, eIN, HILITE_LW, INADMM_COL_GRN, HALO_ON, HALO_COL, HALO_W_EXTRA);
        end
        mark_st_map(ax, Xc, Yc, start_node, end_node, 90, 18);
        title(ax, sprintf('%s — InADMM edges |z|\\ge %.2g  (|V|=%d, |E|=%d)', fileList{f}, thr_abs, n, m), 'Color', [1 1 1]);
    else
        p = base_plot_plain(G, Xc, Yc, NODE_DOT_SZ, BASE_EDGE_COL, BASE_EDGE_ALPHA, BASE_EDGE_W);
        if ~isempty(eIN)
            highlight_edges_plain(gca, Xc, Yc, sIdx, tIdx, eIN, HILITE_LW, INADMM_COL_GRN, HALO_ON, HALO_COL, HALO_W_EXTRA);
        end
        mark_st_plain(Xc, Yc, start_node, end_node, 90, 18);
        title(sprintf('%s — InADMM edges |z|\\ge %.2g  (|V|=%d, |E|=%d)', fileList{f}, thr_abs, n, m));
    end

    % -------- Convergence plots --------
    plot_history(histIN, sprintf('InADMM (direct) — %s', fileList{f}));
end

%% =================== Solver (implicit, fast direct chol) ===================
function [z, a, v, hist] = inadmm_lasso_weighted_direct_implicit(D, sIdx, tIdx, w, y, lambda_vec, opts)
% InADMM with exact inner solve via sparse Cholesky of L = D*W^{-2}*D' + rho*I.
% Avoids forming Q; uses identities:
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
        a     = soft_vec(z_hat + v, lamv/rho);   % elementwise λ_j/ρ
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

%% =================== Plot helpers (geo + planar, saturated) ================
function ax = base_plot_map(G, X, Y, sIdx, tIdx, basemapName, baseLW, baseColor, baseAlpha)
    figure('Color','w');
    ax = geoaxes; geobasemap(ax, basemapName); hold(ax, 'on');
    m = numedges(G);
    latE = [Y(sIdx)'; Y(tIdx)'; nan(1,m)]; latE = latE(:);
    lonE = [X(sIdx)'; X(tIdx)'; nan(1,m)]; lonE = lonE(:);
    h = geoplot(ax, latE, lonE, '-', 'LineWidth', baseLW, 'Color', baseColor);
    if baseAlpha<1, h.Color(4) = baseAlpha; end

    % tight viewport
    pad=0.01; latmin=min(Y); latmax=max(Y); lonmin=min(X); lonmax=max(X);
    dlat=max(latmax-latmin,1e-3); dlon=max(lonmax-lonmin,1e-3);
    geolimits(ax,[latmin-pad*dlat, latmax+pad*dlat],[lonmin-pad*dlon, lonmax+pad*dlon]);
    ax.Toolbar.Visible = 'off';
end

function highlight_edges_map(ax, X, Y, sIdx, tIdx, eList, lineW, edgeCol, haloOn, haloCol, haloExtra)
    if isempty(eList), return; end
    k = numel(eList);
    latE = [Y(sIdx(eList))'; Y(tIdx(eList))'; nan(1,k)]; latE = latE(:);
    lonE = [X(sIdx(eList))'; X(tIdx(eList))'; nan(1,k)]; lonE = lonE(:);
    if haloOn
        geoplot(ax, latE, lonE, '-', 'LineWidth', lineW+haloExtra, 'Color', haloCol); hold(ax,'on');
    end
    geoplot(ax, latE, lonE, '-', 'LineWidth', lineW, 'Color', edgeCol); hold(ax,'on');
end

function p = base_plot_plain(G, X, Y, nodeDotSize, baseCol, baseAlpha, baseLW)
    figure('Color','w');
    p = plot(G, 'XData', X, 'YData', Y, ...
        'NodeColor',[0.25 0.25 0.25], 'Marker','.', 'MarkerSize', nodeDotSize, ...
        'EdgeColor', baseCol, 'EdgeAlpha', baseAlpha, 'LineWidth', baseLW);
    axis equal off; hold on;
end

function highlight_edges_plain(ax, X, Y, sIdx, tIdx, eList, lineW, edgeCol, haloOn, haloCol, haloExtra)
    if isempty(eList), return; end
    k = numel(eList);
    XE = [X(sIdx(eList))'; X(tIdx(eList))'; nan(1,k)]; XE = XE(:);
    YE = [Y(sIdx(eList))'; Y(tIdx(eList))'; nan(1,k)]; YE = YE(:);
    if haloOn
        plot(ax, XE, YE, '-', 'LineWidth', lineW+haloExtra, 'Color', haloCol); hold(ax,'on');
    end
    plot(ax, XE, YE, '-', 'LineWidth', lineW, 'Color', edgeCol); hold(ax,'on');
end

function mark_st_map(ax, X, Y, s, t, sz, fs)
    geoscatter(ax, Y([s t]), X([s t]), sz, 'o', 'MarkerEdgeColor','k','MarkerFaceColor','w','LineWidth',1.2);
    text(ax, Y(s), X(s), '  s', 'FontWeight','bold','FontSize',fs,'Color',[1 1 1], ...
        'VerticalAlignment','middle','HorizontalAlignment','left');
    text(ax, Y(t), X(t), '  t', 'FontWeight','bold','FontSize',fs,'Color',[1 1 1], ...
        'VerticalAlignment','middle','HorizontalAlignment','left');
end

function mark_st_plain(X, Y, s, t, sz, fs)
    plot(X(s),Y(s),'ko','MarkerFaceColor','w','MarkerSize',max(7,sqrt(sz)));
    plot(X(t),Y(t),'ko','MarkerFaceColor','w','MarkerSize',max(7,sqrt(sz)));
    text(X(s),Y(s),'  s','FontWeight','bold','FontSize',fs,'Color','k', ...
        'VerticalAlignment','middle','HorizontalAlignment','left');
    text(X(t),Y(t),'  t','FontWeight','bold','FontSize',fs,'Color','k', ...
        'VerticalAlignment','middle','HorizontalAlignment','left');
end

%% =================== Utility helpers ======================================
function [sNode, tNode, info] = pick_left_to_right_nodes(G, Xc, Yc)
    n = numnodes(G);
    cc = conncomp(G); [~, bigComp] = max(accumarray(cc',1)); mask = (cc==bigComp);
    mask = mask(:) ~= 0;
    rx = (Xc - min(Xc)) / max(1e-12, (max(Xc)-min(Xc)));
    rx = rx(:);

    candStart = find(mask);
    q = quantile(rx(mask), 0.15);
    candStart = candStart(rx(candStart) <= q);
    if isempty(candStart), candStart = find(mask); end
    [~, ordL] = sort(rx(candStart), 'ascend');

    picked=false; info = struct('numStartTried',0,'finalDeltaRX',NaN,'numCandsFinal',0);
    for ii=1:numel(ordL)
        sTry = candStart(ordL(ii));
        dAll = full(distances(G, sTry)); dAll = dAll(:);
        reach = isfinite(dAll) & mask;
        if nnz(reach)<=1, continue; end
        deltas = [0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.02 0.00];
        for dd=1:numel(deltas)
            delta = deltas(dd);
            cand = find(reach & (rx >= rx(sTry) + delta)); cand(cand==sTry)=[];
            if isempty(cand), continue; end
            alpha=0.85; beta=0.15;
            sc = alpha*dAll(cand) + beta*(rx(cand) - rx(sTry));
            [~, j] = max(sc); tTry = cand(j);
            if ~isempty(tTry) && isfinite(dAll(tTry)) && tTry~=sTry
                sNode=sTry; tNode=tTry; picked=true;
                info.numStartTried=ii; info.finalDeltaRX=delta; info.numCandsFinal=numel(cand);
                break;
            end
        end
        if picked, break; end
    end
    if ~picked
        for ii=1:numel(ordL)
            sTry = candStart(ordL(ii));
            dAll = full(distances(G, sTry)); dAll = dAll(:);
            reach = find(isfinite(dAll) & mask); reach(reach==sTry)=[];
            if isempty(reach), continue; end
            [~, j] = max(dAll(reach)); tTry = reach(j);
            if ~isempty(tTry) && tTry~=sTry
                sNode=sTry; tNode=tTry; picked=true;
                info.numStartTried=ii; info.finalDeltaRX=-1; info.numCandsFinal=numel(reach);
                break;
            end
        end
    end
    if ~picked, error('Could not pick valid left->right s,t.'); end
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
    phi = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
    h   = abs((phi*43758.5453) - floor(phi*43758.5453));
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

function z = soft_vec(x,tau), z = sign(x).*max(abs(x)-tau,0); end
function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end