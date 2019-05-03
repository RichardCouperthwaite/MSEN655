function [logML] = minFunc4(param)
% This function evaluates the log marginal likelihood of the GP 
% for a given set of hyper-parameters for the data and outputs 
% from the Test2-Model3 structure trained on temperature classes
    x_input = csvread('Data/x_out_Optimization_T2_M3_temp.csv');
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

    x = x_input(trainindex,[1,2,4,5,6,8,9,11,12,13,14,15,16,17,18,19]);
    y = y_input(trainindex);
    
    L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14), param(15), param(16)];
    sn = 0.05;
    sf = param(17);

    [ K, cholL, ~] = GP_Kernel(x, L, sf, sn);
    
    [kd,~] = size(x);
    sumCholLDiag = 0;
    for i=1:kd
        sumCholLDiag = sumCholLDiag + log(cholL(i,i));
    end
    
    calc = -(1/2)*(y.')*((cholL.')\(cholL\y))-sumCholLDiag-(kd/2)*log(2*pi);
    logML = -calc;
    
end
