function [R2] = funcEval1(param)
% This function evaluates the GP for a given set of hyper-parameters for
% the data and outputs from the Final structure trained on temperature
% classes
    x_input = csvread('Data/x_out_Final_temp_opt.csv');
    x_norm1 = x_input - min(x_input(:));
    x_input = x_norm1 ./ max(x_norm1(:));
    y_input = csvread('Data/nporigtemp.csv');
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

    train_x = x_input(trainindex,[2,3,4,5,8,9,11,13,14,17]);
    train_y = y_input(trainindex);
    test_x = x_input(testindex,[2,3,4,5,8,9,11,13,14,17]);
    test_y = y_input(testindex);

    L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10)];

    [~, cholL, ~] = GP_Kernel( train_x, L, param(11), 0.05 );

    alpha_mean = transpose(cholL)\(cholL\train_y);

    [ksd,~] = size(test_x);
    [kd,~] = size(train_x);
    ks = zeros(ksd,kd);
    
    for p=1:ksd;
        for q=1:kd;
            c = 0;
            for i = 1:10
                c = c+ ((test_x(p,i)-train_x(q,i))./L(i)).^2;
            end
            ks(p,q) = (param(11)^2)*exp( -(1/2)*c);
        end;
    end
    
    out = ks*alpha_mean;

    SSR = sum((out - test_y).^2);
    SSE = sum((out - mean(out(:))).^2);
    R2 =  1 - SSR/SSE;
end