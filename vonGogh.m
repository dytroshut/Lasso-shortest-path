%% Revision-Shorest path

clear all
clc
close all 

F = imread('van.png');
F1 = imread('van.png');

% imwrite(F, 'pika.png', 'Transparency', [0 0 0]);
I = rgb2gray(F);
% figure(1)
% imshow(F1)

dF = fLiveWireGetCostFcn(I, 0.5, 0.4, 0.1);
% dF(dF>0.4) = 0.9;
dF(dF<0.4) = 0.1;
figure(2)
imshow(dF);

%
% Def the distance matrix and also the incidence matrix
% assume that every node will connected with its eight neighbor nodes.
% we def the distance as the Euclidean distance of the RGB values.

[weights,D] = Conv_diag(dF);
%%
% We need to modify the weights vector by weights= 2*weights + 1.
%
weights_opt = 100*weights+1; %+rand(size(weights))+1;
% weights_opt = 100*weights+1;


n=size(D,1);
  
% pos = [26,2;52,15];
% pos = [26,2;53,18];   0.5, 0.4, 0.1
% pos = [9,15;52,14];   good
% pos = [7,18;52,14];   ok
% pos = [9,15;52,15];   good

% pos = [9,15;53,16];
% pos = [9,15;53,17];
% pos = [10,14;22,4]; % best up to now

% pos = [7,17;47,8];  % good
% pos = [7,17;58,33];

pos = [7,17;58,33];

start_node = pos(1)+pos(3)*size(dF,2);
end_node = pos(2)+pos(4)*size(dF,2);  


% figure(3)
% imshow(markerF)


%
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


% Plot in the image as the optimal solution
% Since we have the Pa vector, we need to oonvert to a matrix as in 
d1 = size(F,1);
d2 = size(F,2);

[cord_x,cord_y] = NumConvCor(Pa,d2);

pathxy(1:2:2*size(Pa,2))=cord_y';
pathxy(2:2:2*size(Pa,2))=cord_x';
% the above is the pre for intershape of image.
%%
markerF = insertMarker(F1,pos,'plus','color','green','size',1);
OptF = insertShape(markerF,'Line',pathxy, 'Color','red','LineWidth',1);
figure(4)
imshow(OptF)

%% ADMM
lambda_max = norm(D'*Y, 'inf');
lambda = 1e-9*lambda_max; %-9

%[z_cg, history_cg] = lassoInADMM_cg(D, Y, lambda, 1e-8, 1.0); %-8 good
%[z_cg, history_cg] = lassoADMM(D, Y, lambda, 1e-8, 1.0); % fix
%[z_cg, history_cg] = InADMM_initialized(D, Y, lambda, 1e-8, 1.0);

[z_cg, history_cg] = lassoInADMM_cgcount(D, Y, lambda, 1e-8, 1.0);
%%
K3 = length(history_cg.objval);
h3 = figure(5);
plot(1:K3, history_cg.znorm, 'k',...
     1:K3, distance*ones(1,K3), 'k--','LineWidth', 2);
ylabel('Path length'); 
xlabel('iter (k)');

ylim([0 200])
xlim([0 40])
%%
Address_edge = find(z_cg(:,1)~=0);  % Edge address, now need to convert to nodes
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

pathxy = [cord_yBPs cord_xBPs cord_yBPt cord_xBPt];

OptFBP = insertShape(markerF,'Line',pathxy, 'Color','red','LineWidth',1);
p = figure(6);
imshow(OptFBP)

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