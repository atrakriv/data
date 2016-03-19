ts_cnt = 0;
n = 6000;
while n < 6481
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