%% My algorithm lasso
clear all
clc
close all

D = zeros(9,13);
D(1,1) = -1;
D(2,1) = 1;
D(1,2) = -1;
D(3,2) = 1;
D(1,3) = -1;
D(4,3) = 1;
D(2,4) = -1;
D(3,4) = 1;
D(2,5) = -1;
D(5,5) = 1;
D(3,6) = -1;
D(6,6) = 1;
D(4,7) = -1;
D(6,7) = 1;
D(4,8) = -1;
D(7,8) = 1;
D(5,9) = -1;
D(8,9) = 1;
D(6,10) = -1;
D(8,10) = 1;
D(6,11) = -1;
D(9,11) = 1;
D(7,12) = -1;
D(9,12) = 1;
D(8,13) = -1;
D(9,13) = 1;

weight = [3 6 7 1 4 2 3 4 1 1 2 5 2];
W = diag(weight);

Y = zeros(9,1);
Y(1) = 1;
Y(9) = -1;

D = -D*inv(W);

%% main algorithm(initial condition)
tic;
% Input D, Y
% Initial condition.

% while \lambda > 0.000001 
% Not sure about the stopping condition but definitly not equal to 0 consider about error.

% 1.Construct A and InA. Active set and inactive set.
% 2.Calculate a,b, ratio saved.
% 3.Calculate \lambda cross and \lambda join.
% 4.Calculate sign s.

[n,p]= size(D); %n, p dimension of the matrix D.

beta_history = zeros(p,6);
lambda_history = zeros(1,6);
%%
beta = zeros(p,1); %initial beta.
k = 1;
lambda_history(k) = inf;
beta_history(:,k) = beta;

A = [];
ss = [];
DA = [];
lowerbound = 0;
upperbound =0;
lowlit=[];
uplit=[];
tolerance = 0.000001;

%% The empty set is trick to be treated in the first step, 
%  I will calculate the first step seperatly for the sake of simplicty.
InA = [1:p]';
Di = D;
%%
numerator = Di'*Y; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
denumerator = zeros(length(numerator),1);
t_join = check(numerator,denumerator)';
%%
lambda = max(t_join); %**************************************************

pos_jedge = find(abs(t_join-lambda)<0.001);%******************************
A_add = InA(pos_jedge);
%%
a = 0;
b = 0;
DA = 0;
Ds = Add(D,A_add);
s = sign(Ds'*(Y-DA*(a-lambda*b)));
ss = [ss;s];
%%
A = [A;A_add]; % A is a column vector or row vector?
% A = sort(A); % Don't sort in the first place.
DA = Add(D,A); % Add fucntion be used. % Variable A, cosntant D
DAinv = pinv(DA); 
DADA = DA'*DA;
DDinv = pinv(DADA);

%%**********************************************************************%%


%% while loop with the proper breaking condition.
while max(abs(t_join))> tolerance
 k = k+1;
% Compute a b and also the ratio aka the crossing time.
beta = a - lambda*b; %Display beta

a = pinv(DA'*DA)*DA'*Y;
b = pinv(DA'*DA)*ss; % Variable ss 

beta_1 = a-lambda*b;
lambda_history(k) = lambda;
beta_history(A,k) = beta_1;
%% Build the Di matrix
[InA,Di] = DiConstruct(D,A,p); % Cosntructing function be used.

%% Join time variable Di
numerator = Di'*(DA*a-Y);
denumerator = Di'*DA*b;
%% Joining and crossing
t_join = check(numerator,denumerator)'; 
% t_join is aiming to add elements to the active set,
% the targeting set is InA.
t_cross = a./b;
% t_cross is aim to remove elements from the set,
% the targeting set is A.

%% Already make the assumption that we didn't remove elements from A
lambda = max(t_join);  %% t_join can be negative.********************
 
% For indexing, use the set. Here is a problem, how to find it may should
% use equal, more like within a small range.
% Assume that we don't consider about removing elements out.
%% Break when lambda = zero
if lambda < 0.001
a1 = pinv(DA'*DA)*DA'*Y;
b1 = pinv(DA'*DA)*ss;

beta = a1 - lambda*b1; %Display beta
k = k+1;
beta_history(A,k) = beta;
lambda_history(1,k) = lambda; 
break;
end
%% Find the joining elements and then add it to DA(remove from Di)
pos_jedge = find(abs(t_join-lambda)<0.0001); %***************************%
% 0.0001 is just a tolerance but maybe it need to A_addbe changed later.
A_add = InA(pos_jedge)'; %Display A_add
%% Update the sign by using A_add.
% Convert A_add to Ds
Ds = Add(D,A_add);
s = sign(Ds'*(Y-DA*(a-lambda*b)));
ss = [ss;s];


%% Add to the DA 
A = [A;A_add]; % A is a column vector or row vector?
% A = sort(A); % Don't sort in the first place.
DA = Add(D,A); % Add fucntion be used. % Variable A, cosntant D
DAinv = pinv(DA); 
%%
end
toc;


%% plot the regularization path.
% 13 edges, plot beta according to lambda

h = figure(1);
plot(lambda_history,beta_history(1,:),'color','[0 0.4470 0.7410]','LineWidth',2)
hold on
plot(lambda_history,beta_history(2,:))
hold on
plot(lambda_history,beta_history(3,:))
hold on
plot(lambda_history,beta_history(4,:),'color','[0.8500 0.3250 0.0980]','LineWidth',2)
hold on
plot(lambda_history,beta_history(5,:))
hold on
plot(lambda_history,beta_history(6,:),'color','[0.9290 0.6940 0.1250]','LineWidth',2)
hold on
plot(lambda_history,beta_history(7,:))
hold on
plot(lambda_history,beta_history(8,:))
hold on
plot(lambda_history,beta_history(9,:),'color','[0.4940 0.1840 0.5560]','LineWidth',2)
hold on
plot(lambda_history,beta_history(10,:))
hold on
plot(lambda_history,beta_history(11,:),'color','[0.4660 0.6740 0.1880]','LineWidth',2)
hold on
plot(lambda_history,beta_history(12,:))
hold on
plot(lambda_history,beta_history(13,:),'color','[0.3010 0.7450 0.9330]','LineWidth',2)

lamba1 = xline(lambda_history(2),'--r',{'\lambda_1','(6,9)','(8,9)'},'LineWidth',2);
lamba2 = xline(lambda_history(3),'--',{'\lambda_2','(1,2)'},'LineWidth',3,'Color',[0.5 0 0.5]);
lamba3 = xline(lambda_history(4),'--b',{'\lambda_3','(2,3)','(5,8)'},'LineWidth',4);
lamba4 = xline(lambda_history(5),'--',{'\lambda_4','(3,6)'},'LineWidth',5,'Color',[0 0.2 0]);
lamba5 = xline(0,'--','\lambda_5','LineWidth',6,'Color',[0.6350 0.0780 0.1840]);

lamba1.FontSize = 15;
lamba1.LabelOrientation = 'horizontal';
lamba2.FontSize = 15;
lamba2.LabelOrientation = 'horizontal';
lamba3.FontSize = 15;
lamba3.LabelOrientation = 'horizontal';
lamba4.FontSize = 15;
lamba4.LabelOrientation = 'horizontal';
lamba5.FontSize = 15;
lamba5.LabelOrientation = 'horizontal';


set(gca, 'XDir','reverse')
string = {'(1,2)', '(2,3)','(3,6)','(6,9)','(8,9)','(5,8)'};
xt = [0.05  0.05  0.05   0.05  0.05  0.1 ];
yt = [2.85  1.0   1.95  2.15   0.3  0.15 ];
text(xt,yt,string,'FontSize',15)
xlabel('\lambda','FontSize',20)
ylim([0,3])
ylabel('Value of \beta(i)','FontSize',20)
set(gca,'fontsize',15)
hold off








%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DA = Add(D,A) % name another function add to help us build the DA matrix.
DA = [];
% A = sort(A); %% lets not worry about that, in other words, 

limit = length(A);

for i =1:limit
    
DA = [DA, D(:,A(i))];

end
end

%% DiConstruct Function

function [InA,Di] = DiConstruct(D,A,p)

Di = [];

Original = 1:p;

if A == 0
   Di = D;
   InA = Original;
   
else
Original(A)=[]; % Dimension should match
InA = Original;
limit = length(InA);

for i =1:limit
    
Di = [Di, D(:,InA(i))];

end
end

end

%% tjoin up and down function

% function tjoin = check(a,b)
% 
% limit = length(a); %limit also should be equal to length(b), so there is a constraint of a,b.
% bminus = b-1;
% bplus = b+1;
% for i = 1:limit
%     if sign(a(i)) ~= sign(bplus(i))
%         tjoin(i) = a(i)/bminus(i);
%     else
%         tjoin(i) = a(i)/bplus(i);
%     end
% end
% 
% end
%% tjoin up and down function
function tjoin = check(a,b)

limit = length(a); %limit also should be equal to length(b), so there is a constraint of a,b.
bminus = b-1;
bplus = b+1;
for i = 1:limit
    if sign(a(i)) ~= sign(bplus(i))
        tjoin(i) = a(i)/bminus(i);
    else 
        if sign(bminus(i)) == sign(bplus(i))
           tjoin(i) = min(a(i)/bminus(i),a(i)/bplus(i));
        else
           tjoin(i) = a(i)/bplus(i); 
        end
    end
end


end

%% Function that can convert D to D
function D = ConvertDD(Drand)

[n,m]=size(Drand);

for i=1:m
    
    loc1=find(Drand(:,i)==1);
    loc2=find(Drand(:,i)==-1);
    if loc1>loc2
       Drand(loc1,i)=-1;
       Drand(loc2,i)=1;
    end


end

D=Drand;

end


