clear all
load('dataSet.mat');   %Load TrainDataSet and TestDataSet

%Preprocessing 2.1
Category_sum = sum(TrainDataTargets,2);
bar([1 2 3 4 5],Cat_sum')  %Graph with segment number on each category
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

%Architecture Study 2.2
%Step3:
net = newff( TrainData, TrainDataTargets, [10] );
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0;
net.trainParam.epochs=500;

net=train(net,TrainData,TrainDataTargets);
    
accuracy=[];    %Create matrices accuracy, precision, recall
precision=[];   %to store output train data
recall=[];      %for each learning combination in the next steps 


TestDataOutput=sim(net,TestData);
[accuracy(1),precision(:,1),recall(:,1)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

%Step4:
%Variations of one hidden layer neurons
%Results stored in columns 2-7

for i=1:6
    net=newff(TrainData,TrainDataTargets,[i*5]);
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(i+1),precision(:,i+1),recall(:,i+1)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
end


%->First hidden layer with 25 neurons

%Variations of second hidden layer neurons
%Results stored in columns 8-13
for i=1:6
    net=newff(TrainData,TrainDataTargets,[25 i*5]);
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(i+7),precision(:,i+7),recall(:,i+7)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
end
 
%->Second hidden layer with 20/25 neurons

%Step5: 
%Choosing 25 and 0 neurons for first and second hidden layer
%Variatons of training fuction used
%Results stored in columns 14-17
training = { 'traingdx' 'trainlm' 'traingd' 'traingda' };
for i = 1:length( training )
    method = char( training( i ) );
    net=newff(TrainData,TrainDataTargets,[25],{},method);
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(i+13),precision(:,i+13),recall(:,i+13)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
end
   
%traingdx

%Step6
%Choosing 25 and 0 neurons for first and second hidden layer
%a) Variatons of output activation fuction used
%Results stored in columns 18-21
activation = {'harldim' 'tansig' 'logsig' 'purelin'};
for i = 1:length( activation )
    method = char( activation( i ) );
    net=newff(TrainData,TrainDataTargets,[25],{'tansig' 'tansig' method},'traingdx');
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(i+17),precision(:,i+17),recall(:,i+17)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
end
    
%tansig
    
%b) Variatons of output learning algorithm used
%Results stored in columns 22-23
learning = {'learngd' 'learngdm'};
for i = 1:length( learning )
    method = char( learning( i ) );
    net=newff(TrainData,TrainDataTargets,[25],{},'traingdx',method);
    net.divideParam.trainRatio=0.8;
    net.divideParam.valRatio=0.2;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(i+21),precision(:,i+21),recall(:,i+21)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
end

%learngdm
    
 %c) Training and without validation set:
 %Results without validation set stored in column 24
    net=newff(TrainData,TrainDataTargets,[25],{},'traingdx');
    net.divideParam.trainRatio=1.0;
    net.divideParam.valRatio=0;
    net.divideParam.testRatio=0;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(24),precision(:,24),recall(:,24)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
            
 %Results with 20% validation set stored in column 25
     net=newff(TrainData,TrainDataTargets,[25],{},'traingdx');
     net.divideParam.trainRatio=0.8;
     net.divideParam.valRatio=0.2;
     net.divideParam.testRatio=0;
     net=train(net,TrainData,TrainDataTargets);
     TestDataOutput=sim(net,TestData);
     [accuracy(25),precision(:,25),recall(:,25)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
            
%d) Training without validation set, setting epoch number near epochs
%used with validation set
%Results stored in column 26
    net=newff(TrainData,TrainDataTargets,[25],{},'traingdx');
    net.divideParam.trainRatio=1.0;
    net.divideParam.valRatio=0;
    net.divideParam.testRatio=0;
    net.trainParam.epochs=160;
    net=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    [accuracy(26),precision(:,26),recall(:,26)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
    
%Train with validation set stopped at 162 epochs
%160 epochs --> accuracy 90.63%
%170 epochs --> accuracy 85.42%

%e) Variatons of learning ratio using traingd, traingdx
    training = { 'traingdx' 'traingd' };
    cnt=1;
    for j=0.05:0.05:0.4
        for i = 1:length( training )
                method = char( training( i ) );
                net=newff(TrainData,TrainDataTargets,[25],{},method);
                net.trainParam.lr=j;
                net.divideParam.trainRatio=0.8;
                net.divideParam.valRatio=0.2;
                net.divideParam.testRatio=0;
                net=train(net,TrainData,TrainDataTargets);
                TestDataOutput=sim(net,TestData);
                [accuracy(cnt+26),precision(:,cnt+26),recall(:,cnt+26)]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets); 
                cnt=cnt+1;
        end
    end
    %Plot traingdx vs traingd - Accuracy
    p1 = [accuracy(27) accuracy(29) accuracy(31) accuracy(33) accuracy(35) accuracy(37) accuracy(39) accuracy(41)];
    p2 = [accuracy(28) accuracy(30) accuracy(32) accuracy(34) accuracy(36) accuracy(38) accuracy(40) accuracy(42)];
    lr = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4];
    figure;
    hold on;
    grid on;
    plot(lr, p1);
    plot(lr, p2, 'm');
    legend('traingdx', 'traingd');
    xlabel('Learning Ratio');
    ylabel('Accuracy');
    hold off;
   
   