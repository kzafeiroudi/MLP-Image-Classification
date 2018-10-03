clear
clc
load('dataSet.mat');

%Preprocessing 2.1 ********************************************************
Category_sum = sum(TrainDataTargets,2);
segments_num = min(Category_sum);    %Find the minimum number of segments used

index = [];               
for i = 1:5           
    index = [ index find( TrainDataTargets( i, : ), segments_num ) ];   %Find the indices of the first #seg_num columns with '1' in each category
end
    
index = index( :, randperm( size( index, 2 ) ) ); %Mix randomly index's columns
TrainDataTargets = TrainDataTargets( :, index );  %Pick only the selected columns
TrainData = TrainData( :, index );                %of TrainDataTargets and TrainData

[ TrainData, ps1 ] = removeconstantrows( TrainData ); %Removes constant rows which has no info to separate each descriptor

%Obtain a linear independant set and remove unimportant coefficients
%Using mapstd to obtain a set with zero mean and unit standard devation
%Then processpca to obtain uncorrelated rows and remove those with
%variation over 0.005
[ TrainData, ps2 ] = mapstd( TrainData );
[ TrainData, ps3 ] = processpca( TrainData, 0.005 );

%Apply changes to TestData
TestData = removeconstantrows( 'apply', TestData, ps1 );
TestData = mapstd( 'apply', TestData, ps2 );
TestData = processpca( 'apply', TestData, ps3 );
%**************************************************************************

lamda = 0.015;  %set weight decay parameter
d = 0.07;       %set threshold value
k = 100;        %set number of epochs to train

t = [];
sumNZ = zeros(k-1,1); %store count of non zero weights for each epoch
error = [];

%initialize neural network
net = newff(TrainData, TrainDataTargets, [30], {'tansig' 'tansig' 'tansig'}, 'traingdx', 'learngdm');
net.trainParam.lr = 0.8;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;
net.trainParam.epochs = 1;    %set train.epochs = 1 to manually train the network via the loop

t(:,1) = getwb(net);    %store in each column the weights before each epoch

for i = 2:k
    [net, tr] = train(net, TrainData, TrainDataTargets);
    
    t(:, i) = getwb(net);
    t(:, i) = t(:,i) - lamda.*t(:, i-1);
    
    for j = 1:size(t, 1)
        if (abs(t(j, i)) < d)
            t(j, i) = 0;
        end
        if (t(j, i) ~= 0)
            sumNZ(i-1) = sumNZ(i-1) + 1;
        end
    end
 
    error(i-1) = tr.best_perf;  %get the mean square error for one epoch
                                %and store it into error matrix to plot        
    net = setwb(net, t(:, i));   %update weights
end

%Simulation using TestData
TestDataOutput = sim(net, TestData);
[accuracy, precision, recall] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);

%Non Zero Weights plot
figure;
hold on;
title('Non Zero Weights');
xlabel('Epochs');
ylabel('Sum of NZW');
bar(sumNZ');
hold off;
%Mean Square Error plot
figure;
hold on;
grid on;
title('Mean Square Error');
xlabel('Epochs');
ylabel('Mean Square Error');
plot(error, 'r');
hold off;