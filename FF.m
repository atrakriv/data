clear;
clc;
close all;

% labels + 14 features
data = xlsread('pac1');                     %write
feat_num = 14;

%%%% PREPRATION

% To generate same size classes, data is
% sorted based on column 1 (labeles) 
data_sorted=sortrows(data,1);               %write

data_big = [];

c1 = find(data(:,1)==1); sc1 =size(c1,1);
c2 = find(data(:,1)==2); sc2 =size(c2,1);
c3 = find(data(:,1)==3); sc3 =size(c3,1);
c4 = find(data(:,1)==4); sc4 =size(c4,1);


for i=1:8
    data_big = [data_big;data_sorted(sc1+1:sc1+sc2,:);... class 2
                data_sorted(sc1+sc2+1:sc1+sc2+sc3,:);...  class 3
                data_sorted(sc1+sc2+1:sc1+sc2+sc3,:);...  class 3
                data_sorted(sc1+sc2+1:sc1+sc2+1+42,:);... class 3
                data_sorted(sc1+sc2+sc3+1:sc1+sc2+sc3+sc4,:)];
end

data_big = [data;data_big];
kk1 = find(data_big(:,1)==1);   %%% look at the size of class 1 
kk2 = find(data_big(:,1)==2);   %%% look at the size of class 2
kk3 = find(data_big(:,1)==3);   %%% look at the size of class 3
kk4 = find(data_big(:,1)==4);   %%% look at the size of class 4


%%% taking mean out and normalizing
for i = 2 : feat_num + 1
    data_big(:,i) = (data_big(:,i)- mean(data_big(:,i)))/max(data_big(:,i));
end

%%% shuffle data
data_big_shuffled = data_big(randperm(size(data_big,1)),:);   %write

%%% TRAINING

I = 14;
J = 20;
K = 20;
L = 4;

tr_num = 6000;   %%% number of trained data
ts_num = 481;    %%% number of test data

error = zeros(1,tr_num-1);

WJI = rand(J,I+1);                 % 20*15
WKJ = rand(K,J+1);                 % 20*21
WLK = rand(L,K+1);                 % 4*21

dWJI = rand(J,I+1);                % 20*15
dWKJ = rand(K,J+1);                % 20*21
dWLK = rand(L,K+1);                % 4*21

rng('default');
rng(1);

index = 4;
FEL(index-3:index-1)=zeros(1,3) ;

tr_cnt = 0;     %%% training correct recognition counter (last epoch)
ts_cnt = 0;     %%% testing correct recognition counter
epch_num = 6;   %%% number of epochs

for epoch = 1:epch_num
    n = 1;
    while n < tr_num
        XI = data_big_shuffled(n,2:size(data,2))';
        YI = [XI;1];                  % 3*1
        
        D = de2bi(2^(data_big_shuffled(n,1)-1),4);
        D = D' ;
        vJ = WJI*YI;        % 20*1
        XJ = tanh(vJ);      % 20*1
        YJ = [XJ;1];        % 21*1
        
        vK = WKJ*YJ;        % 20*1
        XK = tanh(vK);      % change the tanh coefficient
        YK = [XK;1];
        
        YL = WLK*YK;        % since there is no phi in output layer vL ~ YL
        
         %%%% ***** learning rate used in each 1/3 epoch
        et = [0.038, 0.049, 0.051, 0.059, 0.05, 0.045, 0.05, 0.05, 0.035 ...
              0.02, 0.01, 0.01, 0.005, 0.005, 0.004,0.0009, 0.0008, 0.0008];
        
        id = 3*(epoch-1)+floor(n/2000)+1;
        eta = et(id);
        etaLK = eta;
        
        EL = (D-YL);               % 4*1
        
        FEL(:,index)=norm(EL);         % Error signal history
        
        dWLK = EL*YK'*etaLK;           % 4*21   
        
        dWLK_local = EL*YK'*etaLK;     % 1*21  
        WLK = WLK + dWLK;
                                       % kheili fargh nemikone WKj update
                                       % beshe bad wij to ye cycle ya na
        
        b = 1; a = 1;
        % delK = (1-(tanh(b*vK)).^2).*(WLK(:,1:K)'*EL) ;   %write
        delK = diag(1-(tanh(b*vK)).^2)*(WLK(:,1:K)'*EL) ;   %write
        
        etaKJ = eta;
        dWKJ =  etaKJ*delK*YJ';
        dWKJ_local =  etaKJ*delK*YJ';
        WKJ = WKJ + dWKJ;
        
        
%        del7=diag(Phip7)*e7;
%        del6=diag(Phip6)*w76(:,1:m6)'*del7;
%        del5=diag(Phip5)*w65(:,1:m5)'*del6;
        
        delJ = diag(1-(tanh(b*vJ)).^2)*WKJ(:,1:J)'*delK;
        etaJI = eta;
        dWJI =  etaJI*delJ*YI';
        dWJI_local =  etaJI*delJ*YI';
        WJI = WJI + dWJI;   
        
%         if rem(n,1)==0
%            WLK = WLK + dWLK;
%            WKJ = WKJ + dWKJ; 
%            WJI = WJI + dWJI;
%         end
        
        n = n+1;
        Label = [0;0;0;0];
        idx = find(YL==max(YL));
        Label(idx,1) = 1;  %% Label from netword
        
        Ds_YL_NLab =[D YL Label]
        
        if epoch == epch_num
            if isequal(D,Label)
                tr_cnt = tr_cnt+1;
            end
        end
        
        EL2=norm(FEL(index-3:index));    %measure of energy over 2 steps.
        error(n)=EL2 ;
        index=index+1 ;
    end
    
    figure(epoch)
    plot(error(1,2:end))
    xlabel('iteration')
    ylabel('Error')
    
end

err_mean_epoch = mean(error)
rec_rate_train_epoch = tr_cnt/tr_num


%%%% TESTING

while n < tr_num + ts_num
    XI = data_big_shuffled(n,2:size(data,2))';
    YI = [XI;1];        % 3*1
    
    D = de2bi(2^(data_big_shuffled(n,1)-1),4);
    D = D' ;
    vJ = WJI*YI;        % 20*1
    XJ = tanh(vJ);      % 20*1
    YJ = [XJ;1];        % 21*1
    
    vK = WKJ*YJ;        % 20*1
    XK = tanh(vK);      % change the tanh coefficient
    YK = [XK;1];
       
    YL = WLK*YK;        % since there is no phi in output layer vL ~ YL
     
    n = n+1;
    Label = [0;0;0;0];
    idx = find(YL==max(YL));
    Label(idx,1) = 1;

    Ds_YL_NLab =[D YL Label]
    
    if isequal(D,Label)
        ts_cnt = ts_cnt+1;
    end
end
rec_rate_test = ts_cnt/ts_num
