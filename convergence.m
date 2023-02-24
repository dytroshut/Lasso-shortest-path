 %% Comparision vanGogh ADMM InADMM BasisPursuit Lasso_sqrt

%% Revision-Shorest path

clear all
close all 
clc

F = imread('van.png');

I = rgb2gray(F);
dF = fLiveWireGetCostFcn(I, 0.5, 0.4, 0.1);
dF(dF<0.4) = 0.1;
figure(2)
imshow(dF);

[weights,D] = Conv_diag(dF);
%%
% We need to modify the weights vector 
weights_opt = 100*weights+1;

n=size(D,1);
pos = [7,17;58,33];

start_node = pos(1)+pos(3)*size(dF,2);
end_node = pos(2)+pos(4)*size(dF,2);  
Y = zeros(n,1);
Y(start_node,1) = 1;
Y(end_node) = -1;

plotstst=[];
for i= 1:1:size(D,2)
    [s2,~] = find( D(:,i) == 1 );
    [t2,~] = find( D(:,i) == -1 );
    edgest = [s2, t2];
    plotstst = [plotstst; edgest];
end
vs=plotstst(:,1)';
vt=plotstst(:,2)';


% Weight matrix and weight vector 
W = diag(1./weights_opt);
D = D*W;

% First: Solve by Dijkstra
G2 = graph(vs,vt,weights_opt);
[Pa,distance] = shortestpath(G2,start_node,end_node);

%% ADMM
lambda_max = norm(D'*Y, 'inf');
lambda = 1e-9*lambda_max; %-9  lsqr-3,8e-4

D_sparse = sparse(D);
[z_in, history_in] = lassoInADMM_cg(D, Y, lambda, 1e-8, 1.0); %-8 good %3.003
[z_ad, history_ad] = lassoADMM(D, Y, lambda, 1e-8, 1.0); % fix %4.019sec
[z_bp, history_bp] = basis_pursuit(D, Y, 1, 1);  %35.938 sec
[z_Inp, history_Inp] = InADMM_initialized(D, Y, lambda, 1e-8, 1.0);

%%
x1 = history_in;
x2 = history_ad;
x3 = history_bp;
x4 = history_Inp;

K1 = length(x1.objval);
K2 = length(x2.objval);
K3 = length(x3.objval);
K4 = length(x4.objval);
%%
figure(5);
hold on;
plot(1:K1, x1.znorm,'Color',[0 0.4470 0.7410],'LineWidth', 2)
plot(1:K4, x4.znorm,'Color',[0.4940 0.1840 0.5560],'LineWidth', 2)
plot(1:K2, x2.znorm,'Color',[0.9290 0.6940 0.1250],'LineWidth', 2)
plot(1:K3, x3.znorm,'Color',[0.4660 0.6740 0.1880],'LineWidth', 2)


plot(1:K3, distance*ones(1,K3), 'k--','LineWidth', 2);
ylabel('Path length','FontSize',15); 
xlabel('iter (k)','FontSize',15);
set(gca,'fontsize',15);
legend('InADMM','Initialized InADMM','ADMM','Basis Pursuit','Shortest Path')
ylim([0 200])
xlim([0 40])
hold off;


%% Function plot
function pathxy = pathFinder(z,D,F)
Address_edge = z(:,1)~=0;  % Edge address, now need to convert to nodes
D_address = D(:,Address_edge);

d2 = size(F,2);
plotBP=[];
for i= 1:1:size(D_address,2) 
    [s2,~] = find(D_address(:,i) > 0 );
    [t2,~] = find(D_address(:,i) < 0 );
    edgeBP = [s2, t2];
    plotBP = [plotBP; edgeBP];
end
vsBP=plotBP(:,1)';
vtBP=plotBP(:,2)';

[cord_xBPs,cord_yBPs] = NumConvCor(vsBP,d2);
[cord_xBPt,cord_yBPt] = NumConvCor(vtBP,d2);

pathxy = [cord_yBPs cord_xBPs cord_yBPt cord_xBPt];
end

%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dF = fLiveWireGetCostFcn(dImg, dWz, dWg, dWd)

if nargin < 2,
    dWz = 0.43;
    dWg = 0.43;
    dWd = 0.14;
end

% -------------------------------------------------------------------------
% Calculat the cost function

% The gradient strength cost Fg
dImg = double(dImg);
[dY, dX] = gradient(dImg);
dFg = sqrt(dX.^2 + dY.^2);
dFg = 1 - dFg./max(dFg(:));

% The zero-crossing cost Fz
lFz = ~edge(dImg, 'zerocross',0.02);

% The Sum:

dF = dWz.*double(lFz)+ dWg.*dFg;
% -------------------------------------------------------------------------
end

function [weights,D] = Conv(F)

d1 = size(F,1);  % number of row
d2 = size(F,2);  % number of column

k=1;
num_node = d1*d2;
num_edge = 4*(d1-1)*(d2-1)-(d1-2)*(d2-1)-(d2-2)*(d1-1);

weights = zeros(num_edge,1);                
% Define the incidence matrix D and also the weights vector.

for i = 1:d1  %row
    if i < d1
        for j = 1:d2
           if j < d2    % check the note, full explanation
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j)))^2);
             k=k+1;                    
             D(j+d2*(i-1),k) = 1;
             D(j+d2*(i-1)+1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i,j+1)))^2);
             k=k+1;          
           elseif j==d2      
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1; 
             weights(k) = sqrt(double(F(i,j)-F(i+1,j))^2);
             k=k+1;             
           else 
           end
        end
        
    elseif i == d1   %% bot row
         for j=1:d2-1
           D(j+d2*(i-1),k) = 1;
           D(j+d2*(i-1)+1,k) = -1;
           weights(k) = sqrt((double(F(i,j)-F(i,j+1))^2));
           k=k+1;
         end
    end       
end


end 

function [weights,D] = Conv_diag(F)

d1 = size(F,1);  % number of row
d2 = size(F,2);  % number of column

k=1;
num_node = d1*d2;
num_edge = 6*(d1-1)*(d2-1)-(d1-2)*(d2-1)-(d2-2)*(d1-1);

weights = zeros(num_edge,1);                
% Define the incidence matrix D and also the weights vector.

for i = 1:d1  %row
    if i < d1
        for j = 1:d2
            
           if j==1
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j)))^2);
             k=k+1;
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i+1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j+1)))^2);
             k=k+1;             
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*(i-1)+1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i,j+1)))^2);
             k=k+1;                
               
           elseif (j>1)&&(j < d2)    % check the note, full explanation
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i-1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j-1)))^2);
             k=k+1;                            
               
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j)))^2);
             k=k+1;
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i+1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j+1)))^2);
             k=k+1;             
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*(i-1)+1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i,j+1)))^2);
             k=k+1;          
           elseif j==d2 
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i-1,k) = -1;
             weights(k) = sqrt((double(F(i,j)-F(i+1,j-1)))^2);
             k=k+1;                                          
               
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1; 
             weights(k) = sqrt(double(F(i,j)-F(i+1,j))^2);
             k=k+1;             
           else 
           end
        end
        
    elseif i == d1   %% bot row
         for j=1:d2-1
                        
           D(j+d2*(i-1),k) = 1;
           D(j+d2*(i-1)+1,k) = -1;
           weights(k) = sqrt((double(F(i,j)-F(i,j+1))^2));
           k=k+1;              
         end
    end       
end


end 

%% Function of convert number to coordinates
function [cord_x, cord_y]=NumConvCor(v,d2)
     cord_x = zeros(length(v),1);
     cord_y = zeros(length(v),1);
     
for i=1:length(v)
    
    if rem(v(i),d2)~=0
       cord_x(i) = floor(v(i)/d2)+1; 
       cord_y(i) = rem(v(i),d2);
    else 
       cord_x(i) = floor(v(i)/d2); 
       cord_y(i) = d2;      
    end   
end

end

%% Function of path matrix
function [path] = Pathfunction(theta,D,F)

Address_edge = find(theta~=0);  % Edge address, now need to convert to nodes
D_address = D(:,Address_edge);

d1 = size(F,1);
d2 = size(F,2);

plotBP=[];
for i= 1:1:size(D_address,2)
   
    
    [s2,~] = find( D_address(:,i) > 0 );
    [t2,~] = find( D_address(:,i) < 0 );
    edgeBP = [s2, t2];
    plotBP = [plotBP; edgeBP];
end

vsBP=plotBP(:,1)';
vtBP=plotBP(:,2)';

[cord_xBPs,cord_yBPs] = NumConvCor(vsBP,d2);

[cord_xBPt,cord_yBPt] = NumConvCor(vtBP,d2);

path = [cord_yBPs cord_xBPs cord_yBPt cord_xBPt];

end

function [weights,D] = Conv_diag_big(F)

d1 = size(F,1);  % number of row
d2 = size(F,2);  % number of column

k=1;
num_edge = 6*(d1-1)*(d2-1)-(d1-2)*(d2-1)-(d2-2)*(d1-1);

weights = zeros(num_edge,1);                
% Define the incidence matrix D and also the weights vector.

for i = 1:d1  %row
    if i < d1
        for j = 1:d2
            
           if j==1
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j)))^4;
             k=k+1;
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i+1,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j+1)))^4;
             k=k+1;             
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*(i-1)+1,k) = -1;
             weights(k) = (double(F(i,j)-F(i,j+1)))^4;
             k=k+1;                
               
           elseif (j>1)&&(j < d2)    % check the note, full explanation
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i-1,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j-1)))^4;
             k=k+1;                            
               
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j)))^4;
             k=k+1;
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i+1,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j+1)))^4;
             k=k+1;             
             
             D(j+d2*(i-1),k) = 1;
             D(j+d2*(i-1)+1,k) = -1;
             weights(k) = (double(F(i,j)-F(i,j+1)))^4;
             k=k+1;          
           elseif j==d2 
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i-1,k) = -1;
             weights(k) = (double(F(i,j)-F(i+1,j-1)))^4;
             k=k+1;                                          
               
             D(j+d2*(i-1),k) = 1;
             D(j+d2*i,k) = -1; 
             weights(k) = (double(F(i,j)-F(i+1,j)))^4;
             k=k+1;             
           else 
           end
        end
        
    elseif i == d1   %% bot row
         for j=1:d2-1
                        
           D(j+d2*(i-1),k) = 1;
           D(j+d2*(i-1)+1,k) = -1;
           weights(k) = (double(F(i,j)-F(i,j+1)))^4;
           k=k+1;              
         end
    end       
end


end 