function [R2] = funcEvalSynth(param, x_choice, y_choice, numSynth)
    switch x_choice
        case 1
            x_input = csvread('Data/xFTemp_orig_synthMatch.csv');
            synthx_batch1 = csvread('Data/xFTemp_synth_0.csv');
            synthx_batch2 = csvread('Data/xFTemp_synth_1.csv');
            synthx_batch3 = csvread('Data/xFTemp_synth_2.csv');
            synthx_batch4 = csvread('Data/xFTemp_synth_3.csv');
            synthx_batch5 = csvread('Data/xFTemp_synth_4.csv');
            num1 = 104;
            num2 = 20;
            neuronIndex = [2,3,4,5,8,9,11,13,14,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10)];
            sf = param(11);
        case 2
            x_input = csvread('Data/xFTime_orig_synthMatch.csv');
            synthx_batch1 = csvread('Data/xFTime_synth_0.csv');
            synthx_batch2 = csvread('Data/xFTime_synth_1.csv');
            synthx_batch3 = csvread('Data/xFTime_synth_2.csv');
            synthx_batch4 = csvread('Data/xFTime_synth_3.csv');
            synthx_batch5 = csvread('Data/xFTime_synth_4.csv');
            num1 = 104;
            num2 = 20;
            neuronIndex = [1,2,3,5,6,7,8,9,10,11,13,14,16,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14)];
            sf = param(15);
        case 3
            x_input = csvread('Data/xT1M2_orig_synthMatch.csv');
            synthx_batch1 = csvread('Data/xT1M2_synth_0.csv');
            synthx_batch2 = csvread('Data/xT1M2_synth_1.csv');
            synthx_batch3 = csvread('Data/xT1M2_synth_2.csv');
            synthx_batch4 = csvread('Data/xT1M2_synth_3.csv');
            synthx_batch5 = csvread('Data/xT1M2_synth_4.csv');
            num1 = 104;
            num2 = 20;
            neuronIndex = [1,3,6,9,11,16,17,18,20];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9)];
            sf = param(10);
        case 4
            x_input = csvread('Data/xT2M3_orig_synthMatch.csv');
            synthx_batch1 = csvread('Data/xT2M3_synth_0.csv');
            synthx_batch2 = csvread('Data/xT2M3_synth_1.csv');
            synthx_batch3 = csvread('Data/xT2M3_synth_2.csv');
            synthx_batch4 = csvread('Data/xT2M3_synth_3.csv');
            synthx_batch5 = csvread('Data/xT2M3_synth_4.csv');
            num1 = 104;
            num2 = 20;
            neuronIndex = [1,2,4,5,6,8,9,11,12,13,14,15,16,17,18,19];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8), param(9), param(10), param(11), param(12), param(13), param(14), param(15), param(16)];
            sf = param(17);
        case 5
            x_input = csvread('Data/xT5M4_orig_synthMatch.csv');
            synthx_batch1 = csvread('Data/xT5M4_synth_0.csv');
            synthx_batch2 = csvread('Data/xT5M4_synth_1.csv');
            synthx_batch3 = csvread('Data/xT5M4_synth_2.csv');
            synthx_batch4 = csvread('Data/xT5M4_synth_3.csv');
            synthx_batch5 = csvread('Data/xT5M4_synth_4.csv');
            num1 = 104;
            num2 = 20;
            neuronIndex = [1,3,6,7,10,11,16,17];
            L = [param(1), param(2), param(3), param(4), param(5), param(6), param(7), param(8)];
            sf = param(9);
    end
    switch y_choice
        case 1
            y_input = csvread('Data/Temp_orig_synthMatch.csv');
            synthy_batch1 = csvread('Data/Temp_synth_0.csv');
            synthy_batch2 = csvread('Data/Temp_synth_1.csv');
            synthy_batch3 = csvread('Data/Temp_synth_2.csv');
            synthy_batch4 = csvread('Data/Temp_synth_3.csv');
            synthy_batch5 = csvread('Data/Temp_synth_4.csv');
        case 2
            y_input = csvread('Data/Time_orig_synthMatch.csv');
            synthy_batch1 = csvread('Data/Time_synth_0.csv');
            synthy_batch2 = csvread('Data/Time_synth_1.csv');
            synthy_batch3 = csvread('Data/Time_synth_2.csv');
            synthy_batch4 = csvread('Data/Time_synth_3.csv');
            synthy_batch5 = csvread('Data/Time_synth_4.csv');
            
    end
    switch numSynth
        case 1
            x_input2 = synthx_batch1;
            train_y2 = synthy_batch1;
        case 2
            x_input2 = vertcat(synthx_batch1,synthx_batch2);
            train_y2 = vertcat(synthy_batch1,synthy_batch2);
        case 3
            x_input2 = vertcat(synthx_batch1,synthx_batch2,synthx_batch3);
            train_y2 = vertcat(synthy_batch1,synthy_batch2,synthy_batch3);
        case 4
            x_input2 = vertcat(synthx_batch1,synthx_batch2,synthx_batch3,synthx_batch4);
            train_y2 = vertcat(synthy_batch1,synthy_batch2,synthy_batch3,synthy_batch4);
        case 5
            x_input2 = vertcat(synthx_batch1,synthx_batch2,synthx_batch3,synthx_batch4,synthx_batch5);
            train_y2 = vertcat(synthy_batch1,synthy_batch2,synthy_batch3,synthy_batch4,synthy_batch5);
    end
    x_norm1 = x_input - min(x_input(:));
    x_input = x_norm1 ./ max(x_norm1(:));
    x_norm2 = x_input2 - min(x_input2(:));
    x_input2 = x_norm2 ./ max(x_norm2(:));
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
    train_x2 = x_input2(:,neuronIndex);
    train_x = vertcat(train_x,train_x2);
    train_y = y_input(trainindex); 
    train_y = vertcat(train_y,train_y2);
    
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