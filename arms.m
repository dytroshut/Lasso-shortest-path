
%% Amsterdam shortest path via LASSO (fixed lambda) + Fancy plotting with saturated colors
clear; clc; close all;

%% -------- Load graph ----------
S   = load('ams_1k_graph.mat'); 
fns = fieldnames(S);
G   = S.(fns{1});                          % graph or digraph
if isa(G,'digraph')
    G = graph(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), [], numnodes(G));
end

%% -------- Node coordinates ----------
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

%% -------- Fancy plotting config & geo detection ----------
coordPairUsed = '';
chosenK = [];
for k=1:size(tryNames,1)
    if ismember(tryNames{k,1}, G.Nodes.Properties.VariableNames) && ...
       ismember(tryNames{k,2}, G.Nodes.Properties.VariableNames)
        coordPairUsed = [tryNames{k,1} '/' tryNames{k,2}];
        chosenK = k;
        break;
    end
end
% Decide if coordinates are geographic based on field names
isGeo = false;
if ~isempty(chosenK)
    xname = tryNames{chosenK,1};  yname = tryNames{chosenK,2};
    isGeo = (any(strcmpi(xname, {'lon','longitude'})) && any(strcmpi(yname, {'lat','latitude'})));
end
% Optional heuristic: numbers look like lon/lat ranges
if ~isGeo
    if all(isfinite(Xc)) && all(isfinite(Yc))
        xr = range(Xc); yr = range(Yc);
        if max(abs(Xc))<=180 && max(abs(Yc))<=90 && xr>0 && yr>0 && xr<10 && yr<10
            isGeo = true;  % likely degrees
        end
    end
end
% Manual override if your lon/lat sit in X/Y columns:
FORCE_GEO    = false;          % set true if X/Y are actually lon/lat
if FORCE_GEO, isGeo = true; end

% Dark-map friendly base and saturated highlight colors
USE_BASEMAP    = true;                 % set false to force planar style
BASEMAP_NAME   = 'streets-dark';       % 'streets-dark','satellite','topographic',...
ST_MARKER_SZ   = 90;                   % s/t marker size
ST_FONT_SZ     = 18;                   % s/t label font size

EDGE_BASE_W    = 1.2;                  % base network edge width on map
EDGE_BASE_COL  = [0.90 0.94 1.00];     % bright bluish-gray for dark tiles

% === Saturated path colors ===
SP_COL_YELLOW  = [1.00 0.78 0.00];     % strong yellow (built-in shortest path)
ADMM_COL_RED   = [1.00 0.10 0.10];     % vivid red (ADMM)
INADMM_COL_GRN = [0.00 1.00 0.35];     % vivid green (InADMM)

HALO_ON        = true;                 % subtle halo to separate from tiles
HALO_COL       = [1 1 1]*0.92;         % near-white halo
HALO_W_EXTRA   = 2.0;                  % small halo so color stays saturated

NODE_DOT_SZ    = 8;                    % planar node dot size
HILITE_LW      = 4.2;                  % highlight linewidth (all methods)

%% -------- Edges & base weights ----------
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
    % Euclidean in given coordinates (degrees or planar)
    w0 = hypot(Xc(sIdx) - Xc(tIdx), Yc(sIdx) - Yc(tIdx));
end
w0 = double(w0(:)); w0(~isfinite(w0)) = inf; w0(w0<=0) = eps;

%% -------- Add positive random noise + tiny tie-break ----------
ADD_NOISE  = true;
noise_rel  = 1e-1;      % try 1e-3, 1e-2, 1e-1
rng(123);
if ADD_NOISE
    M   = max(w0(~isinf(w0))); if isempty(M), M = 1; end
    eta = noise_rel * M * rand(m,1);    % positive additive noise
    eta(~isfinite(w0)) = 0;
    w_sel = w0 + eta;
else
    w_sel = w0;
end
w = add_tiebreak(w_sel, sIdx, tIdx, 1e-12);
G.Edges.Weight = w;

%% -------- Pick s (leftmost) and t (robust far-right) ----------
[ start_node, end_node, diagInfo ] = pick_left_to_right_nodes(G, Xc, Yc);
fprintf('Chose s=%d (left), t=%d (right). Tried %d starts; Δrx=%.3f; candidates=%d\n',...
    start_node, end_node, diagInfo.numStartTried, diagInfo.finalDeltaRX, diagInfo.numCandsFinal);

%% -------- Built-in shortest path (reference only) ----------
[pathSP, ~] = shortestpath(G, start_node, end_node, 'Method','auto');
eSP   = findedge(G, pathSP(1:end-1), pathSP(2:end));

%% -------- Build Q = D W^{-1}, y = e_s - e_t ----------
I = [sIdx;  tIdx]; J = [(1:m)'; (1:m)']; V = [ones(m,1); -ones(m,1)];
D = sparse(I,J,V,n,m);
Q = D * spdiags(1./w,0,m,m);
y = sparse(n,1); y(start_node)=1; y(end_node)=-1;

%% -------- Fixed lambda ----------
lambda_max = norm(Q'*y, inf);
lambda     = 1e-4 * lambda_max;

%% -------- Solver options ----------
opts.rho     = 1.0;
opts.alpha   = 1.8;         % over-relaxation
opts.maxit   = 1000;
opts.abstol  = 1e-8;
opts.reltol  = 1e-6;
opts.verbose = false;

% InADMM PCG options
opts.cgtol   = 1e-8;
opts.maxcg   = 2000;

%% -------- Absolute threshold for plotting (no subgraph shortestpath) ---
thr_abs = 2e-3;             % plot edges with |z| >= thr_abs

%% -------- ADMM ----------
[z0,v0,a0] = deal(zeros(m,1));
[zAD, aAD, vAD, histAD] = admm_lasso(Q, y, lambda, opts, z0, a0, v0);
eAD = find(abs(zAD) >= thr_abs);
fprintf('ADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d/%d (noise_rel=%.1e)\n', ...
        max(abs(zAD)), thr_abs, numel(eAD), m, noise_rel);

%% -------- InADMM ----------
[zIN, aIN, vIN, histIN] = inadmm_lasso(Q, y, lambda, opts, z0, a0, v0);
eIN = find(abs(zIN) >= thr_abs);
fprintf('InADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d/%d (noise_rel=%.1e)\n', ...
        max(abs(zIN)), thr_abs, numel(eIN), m, noise_rel);

%% -------- Plots (same style as Athens) ----------
% 1) Built-in SP (strong yellow)
if USE_BASEMAP && isGeo
    ax1 = base_plot_map(G, Xc, Yc, sIdx, tIdx, BASEMAP_NAME, EDGE_BASE_W, EDGE_BASE_COL);
    highlight_edges_map(ax1, Xc, Yc, sIdx, tIdx, eSP, HILITE_LW, SP_COL_YELLOW, HALO_ON, HALO_COL, HALO_W_EXTRA);
    mark_st_map(ax1, Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
else
    p1 = base_plot_plain(G, Xc, Yc, NODE_DOT_SZ);
    highlight_edges_plain(gca, Xc, Yc, sIdx, tIdx, eSP, HILITE_LW, SP_COL_YELLOW, HALO_ON, HALO_COL, HALO_W_EXTRA);
    mark_st_plain(Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
end

% 2) ADMM: edges with |z| >= thr_abs (vivid red)
if USE_BASEMAP && isGeo
    ax2 = base_plot_map(G, Xc, Yc, sIdx, tIdx, BASEMAP_NAME, EDGE_BASE_W, EDGE_BASE_COL);
    if ~isempty(eAD)
        highlight_edges_map(ax2, Xc, Yc, sIdx, tIdx, eAD, HILITE_LW, ADMM_COL_RED, HALO_ON, HALO_COL, HALO_W_EXTRA);
    end
    mark_st_map(ax2, Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
else
    p2 = base_plot_plain(G, Xc, Yc, NODE_DOT_SZ);
    if ~isempty(eAD)
        highlight_edges_plain(gca, Xc, Yc, sIdx, tIdx, eAD, HILITE_LW, ADMM_COL_RED, HALO_ON, HALO_COL, HALO_W_EXTRA);
    end
    mark_st_plain(Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
end

% 3) InADMM: edges with |z| >= thr_abs (vivid green)
if USE_BASEMAP && isGeo
    ax3 = base_plot_map(G, Xc, Yc, sIdx, tIdx, BASEMAP_NAME, EDGE_BASE_W, EDGE_BASE_COL);
    if ~isempty(eIN)
        highlight_edges_map(ax3, Xc, Yc, sIdx, tIdx, eIN, HILITE_LW, INADMM_COL_GRN, HALO_ON, HALO_COL, HALO_W_EXTRA);
    end
    mark_st_map(ax3, Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
else
    p3 = base_plot_plain(G, Xc, Yc, NODE_DOT_SZ);
    if ~isempty(eIN)
        highlight_edges_plain(gca, Xc, Yc, sIdx, tIdx, eIN, HILITE_LW, INADMM_COL_GRN, HALO_ON, HALO_COL, HALO_W_EXTRA);
    end
    mark_st_plain(Xc, Yc, start_node, end_node, ST_MARKER_SZ, ST_FONT_SZ);
end

%% -------- Convergence figures ----------
plot_history(histAD, 'ADMM');
plot_history(histIN, 'InADMM');

%% =================== Helpers ===================

function [sNode, tNode, info] = pick_left_to_right_nodes(G, Xc, Yc)
    % Robust left->right picker with progressive rightness threshold and fallbacks
    n = numnodes(G);
    cc = conncomp(G); [~, bigComp] = max(accumarray(cc',1)); mask = (cc==bigComp);
    mask = mask(:) ~= 0;               % force logical column

    rx = (Xc - min(Xc)) / max(1e-12, (max(Xc)-min(Xc)));
    rx = rx(:);                        % force column

    candStart = find(mask);
    % Prefer leftmost 15% of the component to avoid degenerate starts
    q = quantile(rx(mask), 0.15);
    candStart = candStart(rx(candStart) <= q);
    if isempty(candStart)
        candStart = find(mask);        % fallback: any in component
    end
    [~, ordL] = sort(rx(candStart), 'ascend');

    picked = false; info = struct('numStartTried',0,'finalDeltaRX',NaN,'numCandsFinal',0);

    for ii = 1:numel(ordL)
        sTry = candStart(ordL(ii));
        dAll = distances(G, sTry);     % returns 1×n or n×1
        dAll = full(dAll(:));          % force n×1
        reach = isfinite(dAll) & mask; % n×1 logical

        if nnz(reach)<=1
            continue;
        end

        deltas = [0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.02 0.00];
        for dd = 1:numel(deltas)
            delta = deltas(dd);
            cand = find(reach & (rx >= rx(sTry) + delta));
            % Clamp to valid node ids just in case
            cand = cand(cand>=1 & cand<=n);
            % drop self
            cand(cand==sTry) = [];

            if isempty(cand), continue; end

            alpha=0.85; beta=0.15;     % distance-heavy scoring
            sc = alpha*dAll(cand) + beta*(rx(cand) - rx(sTry));
            [~, j] = max(sc);
            tTry = cand(j);

            if ~isempty(tTry) && isfinite(dAll(tTry)) && tTry~=sTry
                sNode = sTry; tNode = tTry; picked = true;
                info.numStartTried = ii;
                info.finalDeltaRX  = delta;
                info.numCandsFinal = numel(cand);
                break;
            end
        end
        if picked, break; end
    end

    % Ultimate fallback: farthest reachable (any direction) from the leftmost tried start
    if ~picked
        for ii = 1:numel(ordL)
            sTry = candStart(ordL(ii));
            dAll = full(distances(G, sTry)); dAll = dAll(:);
            reach = find(isfinite(dAll) & mask);
            reach(reach==sTry) = [];
            if isempty(reach), continue; end
            [~, j] = max(dAll(reach));
            tTry = reach(j);
            if ~isempty(tTry) && tTry~=sTry
                sNode = sTry; tNode = tTry; picked = true;
                info.numStartTried = ii;
                info.finalDeltaRX  = -1;   % means fallback
                info.numCandsFinal = numel(reach);
                break;
            end
        end
    end

    if ~picked
        error('Could not find valid left->right start/end in the largest component.');
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

function ax = base_plot_map(G, X, Y, sIdx, tIdx, basemapName, baseLW, baseColor)
% Draw graph edges on a geoaxes with a basemap. X=lon, Y=lat (degrees).
    figure('Color','w');
    ax = geoaxes; geobasemap(ax, basemapName); hold(ax, 'on');
    m = numedges(G);

    % NaN-separated edge polylines
    latE = [Y(sIdx)'; Y(tIdx)'; nan(1,m)]; latE = latE(:);
    lonE = [X(sIdx)'; X(tIdx)'; nan(1,m)]; lonE = lonE(:);

    % brighter base network for dark maps
    geoplot(ax, latE, lonE, '-', 'LineWidth', baseLW, 'Color', baseColor);

    % Tight viewport
    pad = 0.01;
    latmin = min(Y); latmax = max(Y);
    lonmin = min(X); lonmax = max(X);
    dlat = (latmax-latmin); dlon = (lonmax-lonmin);
    if dlat==0, dlat=1e-3; end
    if dlon==0, dlon=1e-3; end
    geolimits(ax, [latmin-pad*dlat, latmax+pad*dlat], [lonmin-pad*dlon, lonmax+pad*dlon]);

    ax.Toolbar.Visible = 'off';
end

function highlight_edges_map(ax, X, Y, sIdx, tIdx, eList, lineW, edgeCol, haloOn, haloCol, haloExtra)
% Overlay highlighted edges on geoaxes, with optional halo for contrast.
    if isempty(eList), return; end
    k = numel(eList);
    latE = [Y(sIdx(eList))'; Y(tIdx(eList))'; nan(1,k)]; latE = latE(:);
    lonE = [X(sIdx(eList))'; X(tIdx(eList))'; nan(1,k)]; lonE = lonE(:);

    if haloOn
        geoplot(ax, latE, lonE, '-', 'LineWidth', lineW + haloExtra, 'Color', haloCol); hold(ax,'on');
    end
    geoplot(ax, latE, lonE, '-', 'LineWidth', lineW, 'Color', edgeCol); hold(ax,'on');
end

function mark_st_map(ax, X, Y, s, t, sz, fs)
% Mark s and t on the geoaxes with bold white labels for dark maps.
    geoscatter(ax, Y([s t]), X([s t]), sz, 'o', ...
        'MarkerEdgeColor','k','MarkerFaceColor','w','LineWidth',1.2);
    text(ax, Y(s), X(s), '  s', 'FontWeight','bold', 'FontSize', fs, 'Color',[1 1 1], ...
        'VerticalAlignment','middle', 'HorizontalAlignment','left');
    text(ax, Y(t), X(t), '  t', 'FontWeight','bold', 'FontSize', fs, 'Color',[1 1 1], ...
        'VerticalAlignment','middle', 'HorizontalAlignment','left');
end

function p = base_plot_plain(G, X, Y, nodeDotSize)
% Clean planar style with visible node dots and soft edges.
    figure('Color','w');
    p = plot(G, 'XData', X, 'YData', Y, ...
        'NodeColor',[0.25 0.25 0.25], 'Marker','.', 'MarkerSize', nodeDotSize, ...
        'EdgeColor',[0.65 0.70 0.78], 'EdgeAlpha',0.8, 'LineWidth',1.1);
    axis equal off; hold on;
end

function highlight_edges_plain(ax, X, Y, sIdx, tIdx, eList, lineW, edgeCol, haloOn, haloCol, haloExtra)
% Overlay highlighted edges on a normal axes (planar), with optional halo.
    if isempty(eList), return; end
    k = numel(eList);
    XE = [X(sIdx(eList))'; X(tIdx(eList))'; nan(1,k)]; XE = XE(:);
    YE = [Y(sIdx(eList))'; Y(tIdx(eList))'; nan(1,k)]; YE = YE(:);
    if haloOn
        plot(ax, XE, YE, '-', 'LineWidth', lineW + haloExtra, 'Color', haloCol); hold(ax,'on');
    end
    plot(ax, XE, YE, '-', 'LineWidth', lineW, 'Color', edgeCol); hold(ax,'on');
end

function mark_st_plain(X, Y, s, t, sz, fs)
% Mark s and t on a normal axes.
    plot(X(s), Y(s), 'ko', 'MarkerFaceColor','w', 'MarkerSize', max(7, sqrt(sz)));
    plot(X(t), Y(t), 'ko', 'MarkerFaceColor','w', 'MarkerSize', max(7, sqrt(sz)));
    text(X(s), Y(s), '  s', 'FontWeight','bold', 'FontSize', fs, 'Color', 'k', ...
        'VerticalAlignment','middle', 'HorizontalAlignment','left');
    text(X(t), Y(t), '  t', 'FontWeight','bold', 'FontSize', fs, 'Color', 'k', ...
        'VerticalAlignment','middle', 'HorizontalAlignment','left');
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

%% ======== Solvers (ADMM / InADMM) ========

function [z, a, v, hist] = admm_lasso(Q, y, lambda, opts, z0, a0, v0)
    rho = opts.rho; alpha = getf(opts,'alpha',1.8);
    maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
    [~,m] = size(Q); z=z0; a=a0; v=v0;
    A = (Q'*Q) + rho*speye(m); rhs0 = Q'*y;
    useChol=true; try R = chol(A,'lower'); catch, useChol=false; end
    hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
    for k=1:maxit
        rhs = rhs0 + rho*(a - v);
        if useChol, z = R'\(R\rhs); else, z = A\rhs; end
        z_hat = alpha*z + (1-alpha)*a;
        a_old = a;
        a = soft(z_hat + v, lambda/rho);
        v = v + (z_hat - a);
        r = z - a; s = rho*(a - a_old);
        rnorm=norm(r); snorm=norm(s);
        eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
        eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
        hist.rnorm(k)=rnorm; hist.snorm(k)=snorm; hist.obj(k)=0.5*norm(Q*z - y)^2 + lambda*sum(abs(z));
        hist.iters=k; if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
        if verbose && (mod(k,50)==0 || k==1)
            fprintf('ADMM %4d  r=%.2e s=%.2e  obj=%.4e\n',k,rnorm,snorm,hist.obj(k));
        end
    end
    hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
end

function [z, a, v, hist] = inadmm_lasso(Q, y, lambda, opts, z0, a0, v0)
    rho = opts.rho; alpha=getf(opts,'alpha',1.8);
    maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
    cgtol=opts.cgtol; maxcg=opts.maxcg;
    [~,m]=size(Q); z=z0; a=a0; v=v0; rhs0=Q'*y;
    dPc = full(sum(Q.^2,2) + rho); Mfun = @(x) x ./ dPc;  % diagonal preconditioner
    Aop  = @(x) Q*(Q'*x) + rho*x;                        % (QQ' + rho I)x
    hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
    for k=1:maxit
        h = rhs0 + rho*(a - v); b = Q*h;
        [eta, flag] = pcg(Aop, b, cgtol, maxcg, Mfun, [], []);
        if verbose && flag~=0, fprintf('  PCG flag=%d\n',flag); end
        z = (1/rho)*(h - Q'*eta);
        z_hat= alpha*z + (1-alpha)*a;
        a_old= a;
        a = soft(z_hat + v, lambda/rho);
        v = v + (z_hat - a);
        r = z - a; s = rho*(a - a_old);
        rnorm=norm(r); snorm=norm(s);
        eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
        eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
        hist.rnorm(k)=rnorm; hist.snorm(k)=snorm; hist.obj(k)=0.5*norm(Q*z - y)^2 + lambda*sum(abs(z));
        hist.iters=k; if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
    end
    hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
end

function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end
function z = soft(x,tau), z = sign(x).*max(abs(x)-tau,0); end















% %% Amsterdam shortest path via LASSO (fixed lambda), threshold-only plotting + noise (robust s,t)
% clear; clc; close all;
% 
% %% -------- Load graph ----------
% S   = load('ams_1k_graph.mat'); 
% fns = fieldnames(S);
% G   = S.(fns{1});                          % graph or digraph
% if isa(G,'digraph')
%     G = graph(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), [], numnodes(G));
% end
% 
% %% -------- Node coordinates ----------
% tryNames = {'X','Y'; 'x','y'; 'Lon','Lat'; 'lon','lat'; 'Longitude','Latitude'};
% Xc=[]; Yc=[];
% for k=1:size(tryNames,1)
%     if ismember(tryNames{k,1}, G.Nodes.Properties.VariableNames) && ...
%        ismember(tryNames{k,2}, G.Nodes.Properties.VariableNames)
%         Xc = G.Nodes.(tryNames{k,1}); Yc = G.Nodes.(tryNames{k,2}); break;
%     end
% end
% if isempty(Xc) || isempty(Yc), error('Node coordinates not found.'); end
% if iscell(Xc), Xc = cell2mat(Xc); end
% if iscell(Yc), Yc = cell2mat(Yc); end
% Xc = double(Xc(:)); Yc = double(Yc(:));
% 
% %% -------- Edges & base weights ----------
% EN = G.Edges.EndNodes;
% if iscell(EN) || isstring(EN) || iscategorical(EN)
%     sIdx = findnode(G, EN(:,1));  tIdx = findnode(G, EN(:,2));
% else
%     sIdx = EN(:,1);               tIdx = EN(:,2);
% end
% m = numedges(G); n = numnodes(G);
% 
% if ismember('Weight', G.Edges.Properties.VariableNames)
%     w0 = double(G.Edges.Weight);
% else
%     w0 = hypot(Xc(sIdx) - Xc(tIdx), Yc(sIdx) - Yc(tIdx));
% end
% w0 = double(w0(:)); w0(~isfinite(w0)) = inf; w0(w0<=0) = eps;
% 
% %% -------- Add positive random noise + tiny tie-break ----------
% ADD_NOISE  = true;
% noise_rel  = 1e-1;      % try 1e-3, 1e-2, 1e-1
% rng(123);
% if ADD_NOISE
%     M   = max(w0(~isinf(w0))); if isempty(M), M = 1; end
%     eta = noise_rel * M * rand(m,1);    % positive additive noise
%     eta(~isfinite(w0)) = 0;
%     w_sel = w0 + eta;
% else
%     w_sel = w0;
% end
% w = add_tiebreak(w_sel, sIdx, tIdx, 1e-12);
% G.Edges.Weight = w;
% 
% %% -------- Pick s (leftmost) and t (robust far-right) ----------
% [ start_node, end_node, diagInfo ] = pick_left_to_right_nodes(G, Xc, Yc);
% fprintf('Chose s=%d (left), t=%d (right). Tried %d starts; Δrx=%.3f; candidates=%d\n',...
%     start_node, end_node, diagInfo.numStartTried, diagInfo.finalDeltaRX, diagInfo.numCandsFinal);
% 
% %% -------- Built-in shortest path (reference only) ----------
% [pathSP, ~] = shortestpath(G, start_node, end_node, 'Method','auto');
% eSP   = findedge(G, pathSP(1:end-1), pathSP(2:end));
% 
% %% -------- Build Q = D W^{-1}, y = e_s - e_t ----------
% I = [sIdx;  tIdx]; J = [(1:m)'; (1:m)']; V = [ones(m,1); -ones(m,1)];
% D = sparse(I,J,V,n,m);
% Q = D * spdiags(1./w,0,m,m);
% y = sparse(n,1); y(start_node)=1; y(end_node)=-1;
% 
% %% -------- Fixed lambda ----------
% lambda_max = norm(Q'*y, inf);
% lambda     = 1e-4 * lambda_max;
% 
% %% -------- Solver options ----------
% opts.rho     = 1.0;
% opts.alpha   = 1.8;         % over-relaxation
% opts.maxit   = 1000;
% opts.abstol  = 1e-8;
% opts.reltol  = 1e-6;
% opts.verbose = false;
% 
% % InADMM PCG options
% opts.cgtol   = 1e-8;
% opts.maxcg   = 2000;
% 
% %% -------- Absolute threshold for plotting (no subgraph shortestpath) ---
% thr_abs = 2e-3;             % plot edges with |z| >= thr_abs
% 
% %% -------- ADMM ----------
% [z0,v0,a0] = deal(zeros(m,1));
% [zAD, aAD, vAD, histAD] = admm_lasso(Q, y, lambda, opts, z0, a0, v0);
% eAD = find(abs(zAD) >= thr_abs);
% fprintf('ADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d/%d (noise_rel=%.1e)\n', ...
%         max(abs(zAD)), thr_abs, numel(eAD), m, noise_rel);
% 
% %% -------- InADMM ----------
% [zIN, aIN, vIN, histIN] = inadmm_lasso(Q, y, lambda, opts, z0, a0, v0);
% eIN = find(abs(zIN) >= thr_abs);
% fprintf('InADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d/%d (noise_rel=%.1e)\n', ...
%         max(abs(zIN)), thr_abs, numel(eIN), m, noise_rel);
% 
% %% -------- Plots ----------
% % 1) Built-in SP (black)
% figure('Color','w'); p1 = base_plot(G,Xc,Yc);
% highlight(p1,'Edges',eSP,'LineWidth',3.0,'EdgeColor','k');
% mark_st(G,Xc,Yc,start_node,end_node);
% %title(sprintf('Amsterdam — built-in shortest path (black)  |V|=%d |E|=%d', n, m));
% 
% % 2) ADMM: edges with |z| >= thr_abs (dark red)
% figure('Color','w'); p2 = base_plot(G,Xc,Yc);
% if ~isempty(eAD), highlight(p2,'Edges',eAD,'LineWidth',2.6,'EdgeColor',[0.85 0 0]); end
% mark_st(G,Xc,Yc,start_node,end_node);
% %title(sprintf('Amsterdam — ADMM, edges with |z| \\ge %.4g (noise rel %.1e)', thr_abs, noise_rel));
% 
% % 3) InADMM: edges with |z| >= thr_abs (dark green)
% figure('Color','w'); p3 = base_plot(G,Xc,Yc);
% if ~isempty(eIN), highlight(p3,'Edges',eIN,'LineWidth',2.6,'EdgeColor',[0 0.45 0]); end
% mark_st(G,Xc,Yc,start_node,end_node);
% %title(sprintf('Amsterdam — InADMM, edges with |z| \\ge %.4g (noise rel %.1e)', thr_abs, noise_rel));
% 
% %% -------- Convergence figures ----------
% plot_history(histAD, 'ADMM');
% plot_history(histIN, 'InADMM');
% 
% %% =================== Helpers ===================
% 
% function [sNode, tNode, info] = pick_left_to_right_nodes(G, Xc, Yc)
%     % Robust left->right picker with progressive rightness threshold and fallbacks
%     n = numnodes(G);
%     cc = conncomp(G); [~, bigComp] = max(accumarray(cc',1)); mask = (cc==bigComp);
%     mask = mask(:) ~= 0;               % force logical column
% 
%     rx = (Xc - min(Xc)) / max(1e-12, (max(Xc)-min(Xc)));
%     rx = rx(:);                        % force column
% 
%     candStart = find(mask);
%     % Prefer leftmost 15% of the component to avoid degenerate starts
%     q = quantile(rx(mask), 0.15);
%     candStart = candStart(rx(candStart) <= q);
%     if isempty(candStart)
%         candStart = find(mask);        % fallback: any in component
%     end
%     [~, ordL] = sort(rx(candStart), 'ascend');
% 
%     picked = false; info = struct('numStartTried',0,'finalDeltaRX',NaN,'numCandsFinal',0);
% 
%     for ii = 1:numel(ordL)
%         sTry = candStart(ordL(ii));
%         dAll = distances(G, sTry);     % returns 1×n or n×1
%         dAll = full(dAll(:));          % force n×1
%         reach = isfinite(dAll) & mask; % n×1 logical
% 
%         if nnz(reach)<=1
%             continue;
%         end
% 
%         deltas = [0.35 0.30 0.25 0.20 0.15 0.10 0.05 0.02 0.00];
%         for dd = 1:numel(deltas)
%             delta = deltas(dd);
%             cand = find(reach & (rx >= rx(sTry) + delta));
%             % Clamp to valid node ids just in case
%             cand = cand(cand>=1 & cand<=n);
%             % drop self
%             cand(cand==sTry) = [];
% 
%             if isempty(cand), continue; end
% 
%             alpha=0.85; beta=0.15;     % distance-heavy scoring
%             sc = alpha*dAll(cand) + beta*(rx(cand) - rx(sTry));
%             [~, j] = max(sc);
%             tTry = cand(j);
% 
%             if ~isempty(tTry) && isfinite(dAll(tTry)) && tTry~=sTry
%                 sNode = sTry; tNode = tTry; picked = true;
%                 info.numStartTried = ii;
%                 info.finalDeltaRX  = delta;
%                 info.numCandsFinal = numel(cand);
%                 break;
%             end
%         end
%         if picked, break; end
%     end
% 
%     % Ultimate fallback: farthest reachable (any direction) from the leftmost tried start
%     if ~picked
%         for ii = 1:numel(ordL)
%             sTry = candStart(ordL(ii));
%             dAll = full(distances(G, sTry)); dAll = dAll(:);
%             reach = find(isfinite(dAll) & mask);
%             reach(reach==sTry) = [];
%             if isempty(reach), continue; end
%             [~, j] = max(dAll(reach));
%             tTry = reach(j);
%             if ~isempty(tTry) && tTry~=sTry
%                 sNode = sTry; tNode = tTry; picked = true;
%                 info.numStartTried = ii;
%                 info.finalDeltaRX  = -1;   % means fallback
%                 info.numCandsFinal = numel(reach);
%                 break;
%             end
%         end
%     end
% 
%     if ~picked
%         error('Could not find valid left->right start/end in the largest component.');
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
% function p = base_plot(G,X,Y)
%     p = plot(G,'XData',X,'YData',Y,'NodeColor',[.2 .2 .2],'Marker','none', ...
%         'EdgeColor',[0 0.4470 0.7410],'EdgeAlpha',0.78,'LineWidth',1.25);
%     axis equal off; hold on;
% end
% 
% function mark_st(G,X,Y,s,t)
%     plot(X(s),Y(s),'ko','MarkerFaceColor','w','MarkerSize',7);
%     plot(X(t),Y(t),'ko','MarkerFaceColor','w','MarkerSize',7);
%     text(X(s),Y(s),'  s','FontWeight','bold','FontSize',14);
%     text(X(t),Y(t),'  t','FontWeight','bold','FontSize',14);
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
% %% ======== Solvers (ADMM / InADMM) ========
% 
% function [z, a, v, hist] = admm_lasso(Q, y, lambda, opts, z0, a0, v0)
%     rho = opts.rho; alpha = getf(opts,'alpha',1.8);
%     maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
%     [~,m] = size(Q); z=z0; a=a0; v=v0;
%     A = (Q'*Q) + rho*speye(m); rhs0 = Q'*y;
%     useChol=true; try R = chol(A,'lower'); catch, useChol=false; end
%     hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
%     for k=1:maxit
%         rhs = rhs0 + rho*(a - v);
%         if useChol, z = R'\(R\rhs); else, z = A\rhs; end
%         z_hat = alpha*z + (1-alpha)*a;
%         a_old = a;
%         a = soft(z_hat + v, lambda/rho);
%         v = v + (z_hat - a);
%         r = z - a; s = rho*(a - a_old);
%         rnorm=norm(r); snorm=norm(s);
%         eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
%         eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
%         hist.rnorm(k)=rnorm; hist.snorm(k)=snorm; hist.obj(k)=0.5*norm(Q*z - y)^2 + lambda*sum(abs(z));
%         hist.iters=k; if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
%         if verbose && (mod(k,50)==0 || k==1)
%             fprintf('ADMM %4d  r=%.2e s=%.2e  obj=%.4e\n',k,rnorm,snorm,hist.obj(k));
%         end
%     end
%     hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
% end
% 
% function [z, a, v, hist] = inadmm_lasso(Q, y, lambda, opts, z0, a0, v0)
%     rho = opts.rho; alpha=getf(opts,'alpha',1.8);
%     maxit=opts.maxit; abstol=opts.abstol; reltol=opts.reltol; verbose=opts.verbose;
%     cgtol=opts.cgtol; maxcg=opts.maxcg;
%     [~,m]=size(Q); z=z0; a=a0; v=v0; rhs0=Q'*y;
%     dPc = full(sum(Q.^2,2) + rho); Mfun = @(x) x ./ dPc;  % diagonal preconditioner
%     Aop  = @(x) Q*(Q'*x) + rho*x;                        % (QQ' + rho I)x
%     hist.rnorm=zeros(1,maxit); hist.snorm=zeros(1,maxit); hist.obj=zeros(1,maxit); hist.iters=0;
%     for k=1:maxit
%         h = rhs0 + rho*(a - v); b = Q*h;
%         [eta, flag] = pcg(Aop, b, cgtol, maxcg, Mfun, [], []);
%         if verbose && flag~=0, fprintf('  PCG flag=%d\n',flag); end
%         z = (1/rho)*(h - Q'*eta);
%         z_hat= alpha*z + (1-alpha)*a;
%         a_old= a;
%         a = soft(z_hat + v, lambda/rho);
%         v = v + (z_hat - a);
%         r = z - a; s = rho*(a - a_old);
%         rnorm=norm(r); snorm=norm(s);
%         eps_p = sqrt(m)*abstol + reltol*max(norm(z), norm(a));
%         eps_d = sqrt(m)*abstol + reltol*norm(rho*v);
%         hist.rnorm(k)=rnorm; hist.snorm(k)=snorm; hist.obj(k)=0.5*norm(Q*z - y)^2 + lambda*sum(abs(z));
%         hist.iters=k; if (rnorm<=eps_p)&&(snorm<=eps_d), break; end
%     end
%     hist.rnorm=hist.rnorm(1:hist.iters); hist.snorm=hist.snorm(1:hist.iters); hist.obj=hist.obj(1:hist.iters);
% end
% 
% function val = getf(s,f,def), if isfield(s,f), val=s.(f); else, val=def; end, end
% function z = soft(x,tau), z = sign(x).*max(abs(x)-tau,0); end