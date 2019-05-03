clc;
% This code completes optimization of the Gaussian process Hyper-parameters
% using the output from the five CNN structures selected. The target
% variable is specified by what that particular structure was trained on.
% This code requires 5 minFunc files, 5 funcEval files and the GP_Kernel.m
% file coded by Richard Couperthwaite. 
% The data for for the analysis is saved in several csv files that need to
% be in the same directory.
% This code outputs figures for the log marginal likelihood, R^2 value and
% initial L value for each of the results from 100 random start points.
% The initial L values are equal for all dimensions and defined by a random 
% value between 0.001 and 2


h = waitbar(0,'Optimization Code...');
options = optimoptions(@fminunc, 'Algorithm','quasi-newton');
run = [];
for i = 1:100;
    a = random('unif',0.001,2);
    x0 = [a,a,a,a,a,a,a,a,a,a,1];
    try
        [x,fval,exitflag,output] = fminunc(@minFunc1, x0, options);
        display(x)
        R2 = funcEval1(x)
    catch ME
        display(ME.message)
        x = x0;
        display(x)
        R2 = -1
        fval = 0
    end
    run(i) = i;
    R2plot1(i) = R2;
    Initplot1(i) = a;
    LML1(i) = fval;
    for j = 1:11;
        param1(i,j) = x(j);
    end
    waitbar(i/500)
end

figure(1)
scatter(run,R2plot1)
title('R2 values')
figure(2)
scatter(run,Initplot1)
title('initial Value')
figure(3)
scatter(run,LML1)
title('Log Marginal Likelihood')

run = [];
for i = 1:100;
    a = random('unif',0.001,2);
    x0 = [a,a,a,a,a,a,a,a,a,a,a,a,a,a,1];
    try
        [x,fval,exitflag,output] = fminunc(@minFunc2, x0, options);
        display(x)
        R2 = funcEval2(x)
    catch ME
        display(ME.message)
        display(ME.stack)
        x = x0;
        display(x)
        R2 = -1
    end
    run(i) = i;
    R2plot2(i) = R2;
    Initplot2(i) = a;
    LML2(i) = fval;
    for j = 1:15;
        param2(i,j) = x(j);
    end
    waitbar((i+100)/500)
end

figure(4)
scatter(run,R2plot2)
title('R2 values')
figure(5)
scatter(run,Initplot2)
title('initial Value')
figure(6)
scatter(run,LML2)
title('Log Marginal Likelihood')

run = [];
for i = 1:100;
    a = random('unif',0.001,2);
    x0 = [a,a,a,a,a,a,a,a,a,1];
    try
        [x,fval,exitflag,output] = fminunc(@minFunc3, x0, options);
        display(x)
        R2 = funcEval3(x)
    catch ME
        display(ME.message)
        display(ME.stack)
        x = x0;
        display(x)
        R2 = -1
    end
    run(i) = i;
    R2plot3(i) = R2;
    Initplot3(i) = a;
    LML3(i) = fval;
    for j = 1:10;
        param3(i,j) = x(j);
    end
    waitbar((i+200)/500)
end

figure(7)
scatter(run,R2plot3)
title('R2 values')
figure(8)
scatter(run,Initplot3)
title('initial Value')
figure(9)
scatter(run,LML3)
title('Log Marginal Likelihood')

run = [];
for i = 1:100;
    a = random('unif',0.001,2);
    x0 = [a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,1];
    try
        [x,fval,exitflag,output] = fminunc(@minFunc4, x0, options);
        display(x)
        R2 = funcEval4(x)
    catch ME
        display(ME.message)
        display(ME.stack)
        x = x0;
        display(x)
        R2 = -1
    end
    run(i) = i;
    R2plot4(i) = R2;
    Initplot4(i) = a;
    LML4(i) = fval;
    for j = 1:17;
        param4(i,j) = x(j);
    end
    waitbar((i+300)/500)
end

figure(10)
scatter(run,R2plot4)
title('R2 values')
figure(11)
scatter(run,Initplot4)
title('initial Value')
figure(12)
scatter(run,LML4)
title('Log Marginal Likelihood')

run = [];
for i = 1:100;
    a = random('unif',0.001,2);
    x0 = [a,a,a,a,a,a,a,a,1];
    minFunc5(x0)
    try
        [x,fval,exitflag,output] = fminunc(@minFunc5, x0, options);
        display(x)
        R2 = funcEval5(x)
    catch ME
        display(ME.message)
        display(ME.stack.file)
        display(ME.stack.name)
        display(ME.stack.line)
        x = x0;
        display(x)
        R2 = -1
    end
    run(i) = i;
    R2plot5(i) = R2;
    Initplot5(i) = a;
    LML5(i) = fval;
    for j = 1:9;
        param5(i,j) = x(j);
    end
    waitbar((i+400)/500)
end

figure(13)
scatter(run,R2plot5)
title('R2 values')
figure(14)
scatter(run,Initplot5)
title('initial Value')
figure(15)
scatter(run,LML5)
title('Log Marginal Likelihood')

%% Output the optimal values found

display('###########################################')
display('Optimal Results for structure 1:')
display('Maximum R^2 Value: ')
display(max(R2plot1))
Struc1R2(1) = max(R2plot1);
index = find(R2plot1==max(R2plot1));
display('Hyper-Parameter Values: ');
display(param1(index,:))
L1 = param1(index,:);
display('###########################################')
display('Optimal Results for structure 2:')
display('Maximum R^2 Value: ')
display(max(R2plot2))
Struc2R2(1) = max(R2plot2);
index = find(R2plot2==max(R2plot2));
display('Hyper-Parameter Values: ');
display(param2(index,:))
L2 = param2(index,:);
display('###########################################')
display('Optimal Results for structure 3:')
display('Maximum R^2 Value: ')
display(max(R2plot3))
Struc2R3(1) = max(R2plot3);
index = find(R2plot3==max(R2plot3));
display('Hyper-Parameter Values: ');
display(param3(index,:))
L3 = param3(index,:);
display('###########################################')
display('Optimal Results for structure 4:')
display('Maximum R^2 Value: ')
display(max(R2plot4))
Struc4R2(1) = max(R2plot4);
index = find(R2plot4==max(R2plot4));
display('Hyper-Parameter Values: ');
display(param4(index,:))
L4 = param4(index,:);
display('###########################################')
display('Optimal Results for structure 5:')
display('Maximum R^2 Value: ')
display(max(R2plot5))
Struc5R2(1) = max(R2plot5);
index = find(R2plot5==max(R2plot5));
display('Hyper-Parameter Values: ');
display(param5(index,:))
L5 = param5(index,:);


%% #######################################################################
% This section of the code does the analysis with a smaller subsection of
% the data determined by high and low magnification values. The
% microns/pixel value of each image has been calculated and the images with
% >1.5 microns/pixel are put in one group and the images with <0.05
% microns/pixel are placed in a second group. The analysis is then done
% while only training and testing with these two groups.

highR2_1 = funcEval(L1,1,1);
display('R^2 for high mag for structure 1:')
display(highR2_1)
highR2_2 = funcEval(L2,2,2);
display('R^2 for high mag for structure 2:')
display(highR2_2)
highR2_3 = funcEval(L3,3,2);
display('R^2 for high mag for structure 3:')
display(highR2_3)
highR2_4 = funcEval(L4,4,1);
display('R^2 for high mag for structure 4:')
display(highR2_4)
highR2_5 = funcEval(L5,5,2);
display('R^2 for high mag for structure 5:')
display(highR2_5)

lowR2_1 = funcEval(L1,6,3);
display('R^2 for low mag for structure 1:')
display(lowR2_1)
lowR2_2 = funcEval(L2,7,4);
display('R^2 for low mag for structure 2:')
display(lowR2_2)
lowR2_3 = funcEval(L3,8,4);
display('R^2 for low mag for structure 3:')
display(lowR2_3)
lowR2_4 = funcEval(L4,9,3);
display('R^2 for low mag for structure 4:')
display(lowR2_4)
lowR2_5 = funcEval(L5,10,4);
display('R^2 for low mag for structure 5:')
display(lowR2_5)


%% #######################################################################
% this test measures the effectiveness of adding synthetic images in
% various percentages to the original data. Only comparitive images have
% been used, and the synthetic images are only appended to the training set
% 1 batch of synthetic data is equivalent to 25% of the original data


R2 = funcEvalSynth(L1, 1, 1, 1);
display('R^2 value for evaluation for structure 1 with 1 batch of synthetic data added')
display(R2)
Struc1R2(2) = R2;
R2 = funcEvalSynth(L1, 1, 1, 2);
display('R^2 value for evaluation for structure 1 with 2 batches of synthetic data added')
display(R2)
Struc1R2(3) = R2;
R2 = funcEvalSynth(L1, 1, 1, 3);
display('R^2 value for evaluation for structure 1 with 3 batches of synthetic data added')
display(R2)
Struc1R2(4) = R2;
R2 = funcEvalSynth(L1, 1, 1, 4);
display('R^2 value for evaluation for structure 1 with 4 batches of synthetic data added')
display(R2)
Struc1R2(5) = R2;
R2 = funcEvalSynth(L1, 1, 1, 5);
display('R^2 value for evaluation for structure 1 with 5 batches of synthetic data added')
display(R2)
Struc1R2(6) = R2;

R2 = funcEvalSynth(L2, 2, 2, 1);
display('R^2 value for evaluation for structure 2 with 1 batch of synthetic data added')
display(R2)
Struc2R2(2) = R2;
R2 = funcEvalSynth(L2, 2, 2, 2);
display('R^2 value for evaluation for structure 2 with 2 batches of synthetic data added')
display(R2)
Struc2R2(3) = R2;
R2 = funcEvalSynth(L2, 2, 2, 3);
display('R^2 value for evaluation for structure 2 with 3 batches of synthetic data added')
display(R2)
Struc2R2(4) = R2;
R2 = funcEvalSynth(L2, 2, 2, 4);
display('R^2 value for evaluation for structure 2 with 4 batches of synthetic data added')
display(R2)
Struc2R2(5) = R2;
R2 = funcEvalSynth(L2, 2, 2, 5);
display('R^2 value for evaluation for structure 2 with 5 batches of synthetic data added')
display(R2)
Struc2R2(6) = R2;

R2 = funcEvalSynth(L3, 3, 2, 1);
display('R^2 value for evaluation for structure 3 with 1 batch of synthetic data added')
display(R2)
Struc3R2(2) = R2;
R2 = funcEvalSynth(L3, 3, 2, 2);
display('R^2 value for evaluation for structure 3 with 2 batches of synthetic data added')
display(R2)
Struc3R2(3) = R2;
R2 = funcEvalSynth(L3, 3, 2, 3);
display('R^2 value for evaluation for structure 3 with 3 batches of synthetic data added')
display(R2)
Struc3R2(4) = R2;
R2 = funcEvalSynth(L3, 3, 2, 4);
display('R^2 value for evaluation for structure 3 with 4 batches of synthetic data added')
display(R2)
Struc3R2(5) = R2;
R2 = funcEvalSynth(L3, 3, 2, 5);
display('R^2 value for evaluation for structure 3 with 5 batches of synthetic data added')
display(R2)
Struc3R2(6) = R2;

R2 = funcEvalSynth(L4, 4, 1, 1);
display('R^2 value for evaluation for structure 4 with 1 batch of synthetic data added')
display(R2)
Struc4R2(2) = R2;
R2 = funcEvalSynth(L4, 4, 1, 2);
display('R^2 value for evaluation for structure 4 with 2 batches of synthetic data added')
display(R2)
Struc4R2(3) = R2;
R2 = funcEvalSynth(L4, 4, 1, 3);
display('R^2 value for evaluation for structure 4 with 3 batches of synthetic data added')
display(R2)
Struc4R2(4) = R2;
R2 = funcEvalSynth(L4, 4, 1, 4);
display('R^2 value for evaluation for structure 4 with 4 batches of synthetic data added')
display(R2)
Struc4R2(5) = R2;
R2 = funcEvalSynth(L4, 4, 1, 5);
display('R^2 value for evaluation for structure 4 with 5 batches of synthetic data added')
display(R2)
Struc4R2(6) = R2;

R2 = funcEvalSynth(L5, 5, 2, 1);
display('R^2 value for evaluation for structure 5 with 1 batch of synthetic data added')
display(R2)
Struc5R2(2) = R2;
R2 = funcEvalSynth(L5, 5, 2, 2);
display('R^2 value for evaluation for structure 5 with 2 batches of synthetic data added')
display(R2)
Struc5R2(3) = R2;
R2 = funcEvalSynth(L5, 5, 2, 3);
display('R^2 value for evaluation for structure 5 with 3 batches of synthetic data added')
display(R2)
Struc5R2(4) = R2;
R2 = funcEvalSynth(L5, 5, 2, 4);
display('R^2 value for evaluation for structure 5 with 4 batches of synthetic data added')
display(R2)
Struc5R2(5) = R2;
R2 = funcEvalSynth(L5, 5, 2, 5);
display('R^2 value for evaluation for structure 5 with 5 batches of synthetic data added')
display(R2)
Struc5R2(6) = R2;


% Output the plot of the effect of synthetic data on the 

x = [0 1 2 3 4 5];
plot(x,Struc1R2,'LineStyle','--','LineWidth',0.5,'Marker','.','MarkerSize',12); hold on
plot(x,Struc2R2,'LineStyle','--','LineWidth',0.5,'Marker','.','MarkerSize',12);
plot(x,Struc3R2,'LineStyle','--','LineWidth',0.5,'Marker','.','MarkerSize',12);
plot(x,Struc4R2,'LineStyle','--','LineWidth',0.5,'Marker','.','MarkerSize',12);
plot(x,Struc5R2,'LineStyle','--','LineWidth',0.5,'Marker','.','MarkerSize',12);
legend('Final-Temp','Final-Time','Test1-Model2','Test2-Model3','Test5-Model4')
ylim([0,1])
xlabel('Number of Batches of Synthetic Image Data')
ylabel('Coefficient of Determination')
title({'Effect of using Different amounts of Synthetic Images'; 'on the Coefficient of Determination'})
hold off

