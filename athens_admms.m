%% Athens road graph — built-in shortest path + LASSO (continuation ISTA)
clear; clc; close all;

%% -------- Load graph (robust to variable name inside .mat) ----------
S   = load('athens_1k_graph.mat'); 
fns = fieldnames(S);
G   = S.(fns{1});                          % expect a MATLAB graph or digraph

% If it's a digraph, convert to undirected for this demo
if isa(G,'digraph')
    G = graph(G.Edges.EndNodes(:,1), G.Edges.EndNodes(:,2), [], numnodes(G));
end

%% -------- Get coordinates (try common field names) ----------
coordFields = {'X','Y'; 'x','y'; 'Lon','Lat'; 'lon','lat'; 'Longitude','Latitude'};
Xc = []; Yc = [];
for k = 1:size(coordFields,1)
    if ismember(coordFields{k,1}, G.Nodes.Properties.VariableNames) && ...
       ismember(coordFields{k,2}, G.Nodes.Properties.VariableNames)
        Xc = G.Nodes.(coordFields{k,1});
        Yc = G.Nodes.(coordFields{k,2});
        break
    end
end
if isempty(Xc) || isempty(Yc)
    error('Could not find node coordinate fields (X/Y, x/y, Lon/Lat, etc.).');
end
if iscell(Xc), Xc = cell2mat(Xc); end
if iscell(Yc), Yc = cell2mat(Yc); end
Xc = double(Xc(:)); 
Yc = double(Yc(:));

%% -------- Edge endpoints as numeric indices ----------
EN = G.Edges.EndNodes;
if iscell(EN) || isstring(EN) || iscategorical(EN)
    sIdx = findnode(G, EN(:,1));
    tIdx = findnode(G, EN(:,2));
else
    sIdx = EN(:,1); 
    tIdx = EN(:,2);
end

%% -------- Edge weights (use existing, else Euclidean length) ----------
if ismember('Weight', G.Edges.Properties.VariableNames)
    w = double(G.Edges.Weight);
else
    w = hypot(Xc(sIdx) - Xc(tIdx), Yc(sIdx) - Yc(tIdx));
    G.Edges.Weight = w;
end
w = double(w(:));
w(~isfinite(w)) = inf;    % non-finite weights treated as barriers
w(w <= 0)       = eps;    % avoid zeros

%% -------- Work inside the largest connected component (LCC) ----------
cc = conncomp(G);
[~, bigComp] = max(accumarray(cc',1));
maskLCC = (cc == bigComp);

% Normalize coords for geometric heuristic
rx = (Xc - min(Xc)) / max(1e-12, max(Xc)-min(Xc));
ry = (Yc - min(Yc)) / max(1e-12, max(Yc)-min(Yc));

%% -------- Choose start (top-right) in LCC; end = farthest reachable ----------
candStart = find(maskLCC);
[~, order] = sort(rx(candStart) + ry(candStart), 'descend');

start_node = [];
for ii = 1:numel(order)
    sTry = candStart(order(ii));
    dTry = distances(G, sTry);              % weighted distances
    if nnz(isfinite(dTry) & maskLCC(:)) > 1
        start_node = sTry;
        distAll    = dTry;
        break
    end
end
if isempty(start_node), error('No valid start node found in largest component.'); end
start_node = double(start_node);

cands = find(maskLCC & isfinite(distAll));
cands(cands == start_node) = [];
[~, idx] = max(distAll(cands));
end_node = double(cands(idx));

%% -------- Built-in shortest path (blue graph, red path) ----------
[pathNodes, pathDist] = shortestpath(G, start_node, end_node, 'Method','auto');
if iscell(pathNodes) || isstring(pathNodes) || iscategorical(pathNodes)
    pathIdx = findnode(G, pathNodes);
else
    pathIdx = double(pathNodes(:));
end

figure('Color','w'); 
p = plot(G, 'XData', Xc, 'YData', Yc, ...
    'NodeColor', [0.2 0.2 0.2], 'Marker', 'none', ...
    'EdgeColor', [0 0.4470 0.7410], 'EdgeAlpha', 0.8, 'LineWidth', 1.2);
axis equal off; hold on;

if ~isempty(pathIdx)
    ePath = findedge(G, pathIdx(1:end-1), pathIdx(2:end));
    highlight(p, 'Edges', ePath, 'LineWidth', 3.0, 'EdgeColor', 'r');
    plot(Xc(pathIdx(1)),  Yc(pathIdx(1)),  'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
    plot(Xc(pathIdx(end)),Yc(pathIdx(end)),'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
    text(Xc(start_node), Yc(start_node), '  s', 'FontWeight','bold', 'Color','k', 'FontSize', 12);
    text(Xc(end_node),   Yc(end_node),   '  t', 'FontWeight','bold', 'Color','k', 'FontSize', 12);
    title(sprintf('Athens 1k — shortest path length = %.2f  (|V|=%d, |E|=%d)', ...
          pathDist, numnodes(G), numedges(G)));
else
    title('No path found between selected nodes.');
end

%% =================== LASSO: Q = D W^{-1}, y = e_s - e_t ===================
n    = numnodes(G); 
m    = numedges(G);
I    = [sIdx;  tIdx];
J    = [(1:m)'; (1:m)'];
V    = [ ones(m,1); -ones(m,1)];
Dinc = sparse(I, J, V, n, m);                 % incidence (+1 at source, -1 at target)
Q    = Dinc * spdiags(1./w, 0, m, m);         % design matrix Q = D W^{-1}

Yrhs = sparse(n,1); 
Yrhs(start_node) = 1; 
Yrhs(end_node)   = -1;

lambda_max = norm(Q' * Yrhs, inf);
lam_start  = 0.8 * lambda_max;                % start high (tiny support)
%lam_final  = 1e-6 * lambda_max;               % very small (path-like support)
lam_final  = 1e-8 * lambda_max; 
nStages    = 25;                              
lams       = exp(linspace(log(lam_start), log(lam_final), nStages));

% ISTA step size (Lipschitz const of grad = ||Q||_2^2)
try
    L = (normest(Q))^2;
catch
    L = (svds(Q,1))^2;
end
tstep = 1 / max(L, 1e-12);

% Warm-start continuation ISTA
z = zeros(m,1);
maxit_per_stage = 10000;
for s = 1:nStages
    lam  = lams(s);
    tau  = lam * tstep;                        % soft-threshold level per step
    for k = 1:maxit_per_stage
        g = Q'*(Q*z - Yrhs);                   % gradient of 0.5||Qz - Y||^2
        z = soft_thresh(z - tstep*g, tau);     % ISTA update
    end
    % (Optional) early exit if support already connects s to t
    eSel_try = find(abs(z) > 1e-8 * max(1, full(max(abs(z)))));
    if ~isempty(eSel_try)
        ENsupp = G.Edges.EndNodes(eSel_try,:);
        if iscell(ENsupp) || isstring(ENsupp) || iscategorical(ENsupp)
            sS = findnode(G, ENsupp(:,1));
            tS = findnode(G, ENsupp(:,2));
        else
            sS = ENsupp(:,1);
            tS = ENsupp(:,2);
        end
        Gsupp_try = graph(sS, tS, w(eSel_try), n, 'OmitSelfLoops');
        [path_try, ~] = shortestpath(Gsupp_try, start_node, end_node, 'Method','unweighted');
        if ~isempty(path_try)
            break; % we have an s–t path within support
        end
    end
end
fprintf('Continuation done at lambda = %.3e\n', lams(s));

% Threshold for visualization
thr   = 1e-6 * max(1, full(max(abs(z))));
eSel  = find(abs(z) > thr);

%% -------- Overlay LASSO support (red) on blue base graph ----------
figure('Color','w');
p2 = plot(G, 'XData', Xc, 'YData', Yc, ...
    'NodeColor', [0.2 0.2 0.2], 'Marker', 'none', ...
    'EdgeColor', [0 0.4470 0.7410], 'EdgeAlpha', 0.8, 'LineWidth', 1.2);
axis equal off; hold on;

if ~isempty(eSel)
    highlight(p2, 'Edges', eSel, 'LineWidth', 3.0, 'EdgeColor', 'r');
end
plot(Xc(start_node), Yc(start_node), 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
plot(Xc(end_node),   Yc(end_node),   'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'w');
text(Xc(start_node), Yc(start_node), '  s', 'FontWeight','bold', 'Color','k', 'FontSize', 12);
text(Xc(end_node),   Yc(end_node),   '  t', 'FontWeight','bold', 'Color','k', 'FontSize', 12);
title(sprintf('LASSO support (red) — |supp|=%d, \\lambda=%.1e', numel(eSel), lams(s)));

%% -------- Extract an s–t path within recovered support (if any) ----------
try
    ENsupp = G.Edges.EndNodes(eSel,:);
    if iscell(ENsupp) || isstring(ENsupp) || iscategorical(ENsupp)
        sS = findnode(G, ENsupp(:,1));
        tS = findnode(G, ENsupp(:,2));
    else
        sS = ENsupp(:,1);
        tS = ENsupp(:,2);
    end
    Gsupp = graph(sS, tS, w(eSel), n, 'OmitSelfLoops');
    [pathAD, distAD] = shortestpath(Gsupp, start_node, end_node, 'Method','unweighted');
    if ~isempty(pathAD)
        eIdxAD = findedge(G, pathAD(1:end-1), pathAD(2:end));
        highlight(p2, 'Edges', eIdxAD, 'LineWidth', 3.0, 'EdgeColor', [0.85 0 0]);
        title(sprintf('LASSO-supported path (dark red), length = %.2f  |supp|=%d', ...
              distAD, numel(eSel)));
    end
catch
    % ok if disconnected
end

%% =================== helper ===================
function z = soft_thresh(x, tau)
    % elementwise soft-threshold
    z = sign(x).*max(abs(x)-tau, 0);
end