function [R2] = funcEval5(param)
    x_input = csvread('Data/x_out_Optimization_T5_M4_time.csv');
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

    train_x = x_input(trainindex,[1,3,6,7,10,11,16,17]);
    train_y = y_input(trainindex);
    test_x = x_input(testindex,[1,3,6,7,10,11,16,17]);
    test_y = y_input(testindex);

    L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8)];

    [~, cholL, ~] = GP_Kernel( train_x, L, param(9), 0.05 );

    alpha_mean = transpose(cholL)\(cholL\train_y);

    [ksd,~] = size(test_x);
    [kd,~] = size(train_x);
    ks = zeros(ksd,kd);
    
    for p=1:ksd;
        for q=1:kd;
            c = 0;
            for i = 1:8
                c = c+ ((test_x(p,i)-train_x(q,i))./L(i)).^2;
            end
            ks(p,q) = (param(9)^2)*exp( -(1/2)*c);
        end;
    end
    
    out = ks*alpha_mean;

    SSR = sum((out - test_y).^2);
    SSE = sum((out - mean(out(:))).^2);
    R2 =  1 - SSR/SSE;
end