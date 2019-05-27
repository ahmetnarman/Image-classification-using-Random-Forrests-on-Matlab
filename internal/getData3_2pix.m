function [ data_train, data_query ] = getData( param )

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

close all;
imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
folderName = './Caltech_101/101_ObjectCategories';
classList = dir(folderName);
classList = {classList(3:end).name} % 10 classes

tic

disp('Loading training images...')
% Load Images -> Description (Dense SIFT)
cnt = 1;
if showImg
    figure('Units','normalized','Position',[.05 .1 .4 .9]);
    suptitle('Training image samples');
end

for c = 1:length(classList)
    subFolderName = fullfile(folderName,classList{c});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    imgIdx{c} = randperm(length(imgList));
    imgIdx_tr = imgIdx{c}(1:imgSel(1)); % Training image indexes
    imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); % Testing image indexes

    % The loop for extracting descriptors from the training set
    for i = 1:length(imgIdx_tr)
        I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));

        % Visualise
        if i < 6 & showImg
            subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            drawnow;
        end

        if size(I,3) == 3
            I = rgb2gray(I); % PHOW work on gray scale image
        end

        % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
        [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
    end

end

%% Important part

toc
tic

disp('Building visual codebook...')
% Build visual vocabulary (codebook) for 'Bag-of-Words method'
% Creating the proper data for the random forrest


data = [];
for i = 1:length(classList)
    % Labels are added to the descriptors 
    newData = [cat(2,desc_tr{i,:})' i*ones(length(cat(2,desc_tr{i,:})'),1)];
    % Data matrix is expanded
    data = [data; newData];
end % training data is ready to be sent to the random forrest with labels

% Random forrest Approach
data=data(randi(length(data),[1,100000]),:);

% write your own codes here
trees = growTrees_2pix(data,param);
% Random forrest Approach


toc
tic

disp('Encoding Images...')
% Vector Quantisation

% write your own codes here

% Bag of words representation for training data (run this for each image)
% The BOW histogram has length of the total amount of tree leafs
BOW_tr = zeros(length(classList),length(desc_tr(i,:)),length(trees(1).prob));

for i = 1:length(classList)
    for j = 1:length(desc_tr(i,:))
        label = testTrees_fast_2pix(desc_tr{i,j}',trees);
        BOW_tr(i,j,:) = sum(histc(label,1:length(trees(1).prob))'); % This part may be wrong
    end
end

data_train = BOW_tr;

% Clear unused varibles to save memory
clearvars desc_tr desc_sel



if showImg
figure('Units','normalized','Position',[.05 .1 .4 .9]);
suptitle('Test image samples');
end

toc
tic

disp('Processing testing images...');
cnt = 1;
% Load Images -> Description (Dense SIFT)
for c = 1:length(classList)
    subFolderName = fullfile(folderName,classList{c});
    imgList = dir(fullfile(subFolderName,'*.jpg'));
    imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));

    for i = 1:length(imgIdx_te)
        I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));

        % Visualise
        if i < 6 & showImg
            subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
            imshow(I);
            cnt = cnt+1;
            drawnow;
        end

        if size(I,3) == 3
            I = rgb2gray(I);
        end
        [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);

    end
end
%suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

% Quantisation

% write your own codes here


% Getting the bag of words for testing data
tic

BOW_te = zeros(length(classList),length(desc_te(1,:)),length(trees(1).prob));

for i = 1:length(classList)
    for j = 1:length(desc_te(i,:))
        label = testTrees_fast_2pix(desc_te{i,j}',trees);
        BOW_te(i,j,:) = sum(histc(label,1:length(trees(1).prob))'); % This part may be wrong
    end
end

data_query = BOW_te;

toc

end

