function [R2] = funcEval(param, x_choice, y_choice)
    switch x_choice
        case 1
            x_input = csvread('Data/highmagx1.csv');
            num1 = 524;
            num2 = 50;
            neuronIndex = [2,3,4,5,8,9,11,13,14,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10)];
            sf = param(11);
        case 2
            x_input = csvread('Data/highmagx2.csv');
            num1 = 524;
            num2 = 50;
            neuronIndex = [1,2,3,5,6,7,8,9,10,11,13,14,16,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14)];
            sf = param(15);
        case 3
            x_input = csvread('Data/highmagx3.csv');
            num1 = 524;
            num2 = 50;
            neuronIndex = [1,3,6,9,11,16,17,18,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9)];
            sf = param(10);
        case 4
            x_input = csvread('Data/highmagx4.csv');
            num1 = 524;
            num2 = 50;
            neuronIndex = [1,2,4,5,6,8,9,11,12,13,14,15,16,17,18,19];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14), param(15), param(16)];
            sf = param(17);
        case 5
            x_input = csvread('Data/highmagx5.csv');
            num1 = 524;
            num2 = 50;
            neuronIndex = [1,3,6,7,10,11,16,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8)];
            sf = param(9);
        case 6
            x_input = csvread('Data/lowmagx1.csv');
            num1 = 656;
            num2 = 70;
            neuronIndex = [2,3,4,5,8,9,11,13,14,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10)];
            sf = param(11);
        case 7
            x_input = csvread('Data/lowmagx2.csv');
            num1 = 656;
            num2 = 70;
            neuronIndex = [1,2,3,5,6,7,8,9,10,11,13,14,16,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14)];
            sf = param(15);
        case 8
            x_input = csvread('Data/lowmagx3.csv');
            num1 = 656;
            num2 = 70;
            neuronIndex = [1,3,6,9,11,16,17,18,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9)];
            sf = param(10);
        case 9
            x_input = csvread('Data/lowmagx4.csv');
            num1 = 656;
            num2 = 70;
            neuronIndex = [1,2,4,5,6,8,9,11,12,13,14,15,16,17,18,19];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14), param(15), param(16)];
            sf = param(17);
        case 10
            x_input = csvread('Data/lowmagx5.csv');
            num1 = 656;
            num2 = 70;
            neuronIndex = [1,3,6,7,10,11,16,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8)];
            sf = param(9);
    end
    switch y_choice
        case 1
            y_input = csvread('Data/highmagytemp.csv');
        case 2
            y_input = csvread('Data/highmagytime.csv');
        case 3
            y_input = csvread('Data/lowmagytemp.csv');
        case 4
            y_input = csvread('Data/lowmagytime.csv');
    end
    x_norm1 = x_input - min(x_input(:));
    x_input = x_norm1 ./ max(x_norm1(:));
    testindex = randperm(num1,num2);
    trainindex = [];
    traincount = 1;

    for i = 1:num1;
        if ismember(i,testindex);
            x = 1;
        else
            trainindex(traincount) = i;
            traincount = traincount + 1;
        end       
    end

    train_x = x_input(trainindex,neuronIndex);
    train_y = y_input(trainindex);
    test_x = x_input(testindex,neuronIndex);
    test_y = y_input(testindex);

    [~, cholL, ~] = GP_Kernel( train_x, L, sf, 0.05 );

    alpha_mean = transpose(cholL)\(cholL\train_y);

    [ksd,~] = size(test_x);
    [kd,~] = size(train_x);
    [~,numParam] = size(L);
    numParam = numParam - 1;
    ks = zeros(ksd,kd);

    for p=1:ksd;
        for q=1:kd;
            c = 0;
            for i = 1:numParam
                c = c+ ((test_x(p,i)-train_x(q,i))./L(i)).^2;
            end
            ks(p,q) = (sf^2)*exp( -(1/2)*c);
        end;
    end

    out = ks*alpha_mean;

    SSR = sum((out - test_y).^2);
    SSE = sum((out - mean(out(:))).^2);
    R2 =  1 - SSR/SSE;
end