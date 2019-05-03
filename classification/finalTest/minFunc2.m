function [logML] = minFunc2(param)
    x_input = csvread('Data/x_out_Final_time_opt.csv');
    x_norm1 = x_input - min(x_input(:));
    x_input = x_norm1 ./ max(x_norm1(:));
    y_input = csvread('Data/nporigtime.csv');
    testindex = randperm(3024,300);
    trainindex = [];
    traincount = 1;

    for i = 1:3024;
        if ismember(i,testindex);
            x = 1;
        else
            trainindex(traincount) = i;
            traincount = traincount + 1;
        end       
    end

    x = x_input(trainindex,[1,2,3,5,6,7,8,9,10,11,13,14,16,20]);
    y = y_input(trainindex);
    
    L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14)];
    sn = 0.05;
    sf = param(15);

    [ K, cholL, ~] = GP_Kernel(x, L, sf, sn);
    
    [kd,~] = size(x);
    sumCholLDiag = 0;
    for i=1:kd
        sumCholLDiag = sumCholLDiag + log(cholL(i,i));
    end
    
    calc = -(1/2)*(y.')*((cholL.')\(cholL\y))-sumCholLDiag-(kd/2)*log(2*pi);
    logML = -calc;
    
end
