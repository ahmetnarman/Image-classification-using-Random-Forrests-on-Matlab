%% Initialisation

clear all; close all; 
init; clc;

%% PART 1: Getting the data using k-means

numBins =30;

% K-means Codebook training and BoW generation is done in 'getData3.m'
[data_train,data_query, TrIm, TeIm]=getData2(numBins);

% For histogram visualisation, add corresponding titles, labels
%%
    figure
    subplot(2,1,1)
    bar([reshape(data_train(3,3,:),1,numBins); reshape(data_query(10,4,:),1,numBins)]');
    title('Corresponding BoW histograms')
    legend('Training Image', 'Testing Image')
    subplot (2,2,3)
    imshow(TrIm{3,3})
    title('Training Image')
    subplot (2,2,4)
    imshow(TeIm{10,4})
    title('Testing image')
    
%% PART 2: Reshaping the data for random forrest classification

S = size(data_train);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_train(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end % training data is ready to be sent to the random forrest with labels

data_train2 = data;


S = size(data_query);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_query(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end % testing data is ready to be sent to the random forrest with labels 
data_query2 = data;

%% PART 3 Growing classification trees and testing

disp('Training RF Classification trees...')

tic

param.num = 300;    % Number of trees
param.depth = 6;    % trees depth
param.splitNum = 8; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

tree=growTrees(data_train2,param); % Growing trees with given parameters

toc

%% Part 4: Testing the classification trees

tic
disp('Testing RF classification trees...')
lab=testTrees_fast(data_query2(:,1:end-1),tree); % Not including labels


C=[]; % The class labels after classification
for i=1:150
    [~,idx]=max(sum(tree(1).prob(lab(i,:),:)));
    C=[C idx];
end

conf = confusionmat(data_query2(:,end)',C); % Confusion matrix

A= C==data_query2(:,end)';
Correctness = 100*sum(A)/150; % Percentage Correctness
disp('Classification Accuracy (in percentage): ');
disp(Correctness);
toc

% Confusion matrix plot
figure
image(conf*(100/15))
colorbar
title('Confusion Matrix for axis alligned test')
xlabel('The input image')
ylabel('Classification output')


%% PART 3': Growing classification trees and testing for 2-pixel test

disp('Training RF Classification trees...')

tic

param.num = 300;    % Number of trees
param.depth = 6;    % trees depth
param.splitNum = 8; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

tree_2pix=growTrees_2pix(data_train2,param);
toc

disp('Testing RF classification trees...')
tic
labels_2pix=testTrees_fast_2pix(data_train2(:,1:end-1),tree_2pix); % No labels
toc

%% PART 4' Classifing 2-pixel test
tic
C=[]; % The class labels after classification
for i=1:150
    [~,idx]=max(sum(tree_2pix(1).prob(labels_2pix(i,:),:)));
    C=[C idx];
end
 
conf = confusionmat(data_query2(:,end)',C); % Confusion matrix

A= C==data_query2(:,end)';
Correctness = 100*sum(A)/150; % Percentage Correctness
disp('Classification Accuracy (in percentage): ');
disp(Correctness);
toc

% Confusion matrix plot
figure
image(conf*(100/15))
colorbar
title('Confusion Matrix for 2pix test')
xlabel('The input image')
ylabel('Classification output')

%% PART 5 Getting the data using random trees codebook 

param.num = 6;
param.depth = 9;    % trees depth
param.splitNum = 8; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

% RF Codebook training and BoW generation is done in 'getData3.m'
[data_train3, data_query3] = getData3_2pix(param);

%% PART 6: Reshaping the data for random forrest classification

S = size(data_train3);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_train3(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end 

data_train4 = data; % Reshaped training data


S = size(data_query3);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_query3(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end 

data_query4 = data; % Reshaped testing data


%% PART 7: Growing classification trees and testing

disp('Training RF Classification trees...')

tic

param.num = 300;    % Number of trees
param.depth = 6;    % trees depth
param.splitNum = 3; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

tree2=growTrees_2pix(data_train4,param);

toc
tic

disp('Testing RF classification trees...')
lab=testTrees_fast_2pix(data_query4(:,1:end-1),tree2); % No labels

% PART 8: Classifing

C=[]; % The class labels after classification
for i=1:150
    [~,idx]=max(sum(tree2(1).prob(lab(i,:),:)));
    C=[C idx];
end

conf = confusionmat(data_query4(:,end)',C);

A= C==data_query4(:,end)';
Correctness = 100*sum(A)/150; % Percentage Correctness
% Add a confusion matrix
disp('Classification Accuracy (in percentage): ');
disp(Correctness);
toc

% Confusion matrix plot
figure
heatmap(conf)
xlabel('The input image')
ylabel('Classification output')

