clear;
clc;
close all;

num = xlsread('pac1');                     %write

num = num(1800:2000,:);
kk1 = find(num(:,1)==1);
kk2 = find(num(:,1)==2);
kk3 = find(num(:,1)==3);
kk4 = find(num(:,1)==4);

shuffled = num(randperm(size(num,1)),:);   %write

I = 14;
J = 20;
K = 4;

NS = size(num,1) ;                 % number of samples
PX = zeros(NS,1);
PY = zeros(NS,1);
WJI = rand(J,I+1);                 % 20*15
WKJ = rand(K,J+1);                 % 4*21

dWJI = rand(J,I+1);                % 20*15
dWKJ = rand(K,J+1);                % 4*21

rng('default');
rng(1);

index = 4;
FEK(index-3:index-1)=zeros(1,3) ;
NEp = 5;

for epoch = 1:NEp
    n = 1;
    while n <= size(num,1)
        XI = shuffled(n,2:size(num,2))';
        YI = [XI;1];                  % 3*1
       
        D = de2bi(2^(shuffled(n,1)-1),4);
        D = D' ;
        vJ = WJI*YI;                  % 20*1
        XJ = tanh(vJ);                % 20*1
        YJ = [XJ;1];                  % 21*1
       
        YK = WKJ*YJ;                  % 1*1
       
        et1=1e-1 ; etn=1e-5 ;
        et11=1e-7 ; etn1=1e-10 ;
        %etaKJ=(etn-et1)/(NS-1)*(n-1)+et1 ;
        etaKJ=(etn-et1)/(NS*NEp-1)*((epoch-1)*size(num,1)+n-1)+et1 ;
       
        EK = (D-YK);                  % 1*1
       
        FEK(:,index)=norm(EK);     % Error signal history
       
        dWKJ = EK*YJ'*etaKJ;          % 1*21   etha?
        dWKJ_local = EK*YJ'*etaKJ;     % 1*21   etha?
        %WKJ = WKJ + dWKJ;
                                       % kheili fargh nemikone WKj update
                                       % beshe bad wij to ye cycle ya na
       
       
        b = 1; a = 1;
        delJ = (1-(tanh(b*vJ)).^2).*(WKJ(:,1:J)'*EK) ;   %write
        etaJI=(1e-1-1e-5)*(1-exp(-0.00001*norm(EK))) ;
        dWJI =  etaJI*delJ*YI';
        dWJI_local =  etaJI*delJ*YI';
        %WJI = WJI + dWJI;
       
        if rem(n,1)==0
           WJI = WJI + dWJI;
           WKJ = WKJ + dWKJ;
        end
       
        n = n+1;
        if epoch > 0
            uu =[D YK];
            uu
        end
       
        EK2=norm(FEK(index-3:index));    %measure of energy over 2 steps.
        error(n)=EK2 ;
        %error((epoch-1)*size(num,1)+n)=EK2 ;
        index=index+1 ;
    end
end


figure(1)
plot(error)
xlabel('iteration')
ylabel('Error')

