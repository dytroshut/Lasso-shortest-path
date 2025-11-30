%% Athens shortest path via LASSO (fixed lambda), threshold-only plotting
% Plots:
%   (1) Built-in shortest path (black) for reference
%   (2) ADMM edges with |z| >= thr_abs (dark red)
%   (3) InADMM edges with |z| >= thr_abs (dark green)
clear; clc; close all;

%% -------- Load graph ----------
S   = load('athens_1k_graph.mat'); 
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

%% -------- Edges & weights ----------
EN = G.Edges.EndNodes;
if iscell(EN) || isstring(EN) || iscategorical(EN)
    sIdx = findnode(G, EN(:,1));  tIdx = findnode(G, EN(:,2));
else
    sIdx = EN(:,1);               tIdx = EN(:,2);
end
m = numedges(G); n = numnodes(G);

% Original weights for reporting/overlay
if ismember('Weight', G.Edges.Properties.VariableNames)
    w0 = double(G.Edges.Weight);
else
    w0 = hypot(Xc(sIdx) - Xc(tIdx), Yc(sIdx) - Yc(tIdx));
end
w0 = double(w0(:)); w0(~isfinite(w0)) = inf; w0(w0<=0) = eps;

% Tiny deterministic tie-break so Dijkstra path is unique (selection only)
w  = add_tiebreak(w0, sIdx, tIdx, 1e-10);
G.Edges.Weight = w;

%% -------- Choose s (top-right) and t (farthest reachable) ----------
cc = conncomp(G); [~, bigComp] = max(accumarray(cc',1)); mask = (cc==bigComp);
rx = (Xc - min(Xc))/max(1e-12, max(Xc)-min(Xc));
ry = (Yc - min(Yc))/max(1e-12, max(Yc)-min(Yc));
candStart = find(mask); [~, ord] = sort(rx(candStart)+ry(candStart),'descend');

start_node = [];
for ii=1:numel(ord)
    sTry = candStart(ord(ii));
    dTry = distances(G, sTry);                % weighted by tie-broken w
    if nnz(isfinite(dTry) & mask(:))>1, start_node = sTry; distAll = dTry; break; end
end
if isempty(start_node), error('No valid start node.'); end
cands = find(mask & isfinite(distAll)); cands(cands==start_node)=[];
[~, j] = max(distAll(cands)); end_node = cands(j);

%% -------- Built-in shortest path (for a reference figure only) ----------
[pathSP, ~] = shortestpath(G, start_node, end_node, 'Method','auto');
eSP   = findedge(G, pathSP(1:end-1), pathSP(2:end));

%% -------- Build Q = D W^{-1}, y = e_s - e_t ----------
I = [sIdx;  tIdx]; J = [(1:m)'; (1:m)']; V = [ones(m,1); -ones(m,1)];
D = sparse(I,J,V,n,m);
Q = D * spdiags(1./w,0,m,m);
y = sparse(n,1); y(start_node)=1; y(end_node)=-1;

%% -------- Fixed lambda ----------
lambda_max = norm(Q'*y, inf);
lambda     = 1e-4 * lambda_max;   % you asked for 0.0001 * lambda_max

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
thr_abs = 1.13*1e-3;             % <---- set your |z| cutoff here, e.g., 0.002

%% -------- ADMM at fixed λ ----------
[z0,v0,a0] = deal(zeros(m,1));
[zAD, aAD, vAD, histAD] = admm_lasso(Q, y, lambda, opts, z0, a0, v0);
eAD = find(abs(zAD) >= thr_abs);
fprintf('ADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d of %d\n', max(abs(zAD)), thr_abs, numel(eAD), m);

%% -------- InADMM at fixed λ ----------
[zIN, aIN, vIN, histIN] = inadmm_lasso(Q, y, lambda, opts, z0, a0, v0);
eIN = find(abs(zIN) >= thr_abs);
fprintf('InADMM: max|z|=%.3e, thr_abs=%.3e, selected edges=%d of %d\n', max(abs(zIN)), thr_abs, numel(eIN), m);

%% -------- Plots ----------
% 1) Built-in SP (black)
figure('Color','w'); p1 = base_plot(G,Xc,Yc);
highlight(p1,'Edges',eSP,'LineWidth',3.0,'EdgeColor','k');
mark_st(G,Xc,Yc,start_node,end_node);
title(sprintf('Built-in shortest path (black)  |V|=%d |E|=%d', n, m));

% 2) ADMM: edges with |z| >= thr_abs (dark red)
figure('Color','w'); p2 = base_plot(G,Xc,Yc);
if ~isempty(eAD), highlight(p2,'Edges',eAD,'LineWidth',2.6,'EdgeColor',[0.85 0 0]); end
mark_st(G,Xc,Yc,start_node,end_node);
title(sprintf('ADMM — edges with |z| \\ge %.4g', thr_abs));

% 3) InADMM: edges with |z| >= thr_abs (dark green)
figure('Color','w'); p3 = base_plot(G,Xc,Yc);
if ~isempty(eIN), highlight(p3,'Edges',eIN,'LineWidth',2.6,'EdgeColor',[0 0.45 0]); end
mark_st(G,Xc,Yc,start_node,end_node);
title(sprintf('InADMM — edges with |z| \\ge %.4g', thr_abs));

%% -------- Convergence figures (residuals + objective) ----------
plot_history(histAD, 'ADMM');
plot_history(histIN, 'InADMM');

%% =================== Helpers ===================

function wU = add_tiebreak(w, sIdx, tIdx, rel)
    w = double(w(:));
    M = max(w(~isinf(w))); if isempty(M), M = 1; end
    eps0 = rel * M;
    phi  = sin(double(sIdx)*12.9898 + double(tIdx)*78.233);
    tb   = abs((phi*43758.5453) - floor(phi*43758.5453)); % [0,1)
    wU   = w + eps0*tb;
end

function p = base_plot(G,X,Y)
    p = plot(G,'XData',X,'YData',Y,'NodeColor',[.2 .2 .2],'Marker','none', ...
        'EdgeColor',[0 0.4470 0.7410],'EdgeAlpha',0.78,'LineWidth',1.25);
    axis equal off; hold on;
end

function mark_st(G,X,Y,s,t)
    plot(X(s),Y(s),'ko','MarkerFaceColor','w','MarkerSize',7);
    plot(X(t),Y(t),'ko','MarkerFaceColor','w','MarkerSize',7);
    text(X(s),Y(s),'  s','FontWeight','bold');
    text(X(t),Y(t),'  t','FontWeight','bold');
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

%% ======== Solvers (over-relaxation & PCG preconditioning) ========

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
    [n,m]=size(Q); z=z0; a=a0; v=v0; rhs0=Q'*y;
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