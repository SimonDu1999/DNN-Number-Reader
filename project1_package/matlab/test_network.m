%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network

confusion_M = zeros(10,10);
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    % The actual predict number is the predict_number-1, so it is in the
    % test data set
    [probability, predict_number] = max(P);
    for j=1:100
        confusion_M(ytest(i+j-1), predict_number(j)) = confusion_M(ytest(i+j-1), predict_number(j)) + 1;
    end
end

disp('The confusion matrix is:');
disp(confusion_M);

% testing my own data
img1 = imread("..//my_images/Page1.png");
img1 = imresize(img1, [28 28]);
img1 = transpose(im2double(rgb2gray(img1)));
img1 = 1-reshape(img1, [28*28 1]);

img2 = imread("..//my_images/Page2.png");
img2 = imresize(img2, [28 28]);
img2 = transpose(im2double(rgb2gray(img2)));
img2 = 1-reshape(img2, [28*28 1]);

img3 = imread("..//my_images/Page3.png");
img3 = imresize(img3, [28 28]);
img3 = transpose(im2double(rgb2gray(img3)));
img3 = 1-reshape(img3, [28*28 1]);

img4 = imread("..//my_images/Page4.png");
img4 = imresize(img4, [28 28]);
img4 = transpose(im2double(rgb2gray(img4)));
img4 = 1-reshape(img4, [28*28 1]);

img5 = imread("..//my_images/Page5.png");
img5 = imresize(img5, [28 28]);
img5 = transpose(im2double(rgb2gray(img5)));
img5 = 1-reshape(img5, [28*28 1]);

layers{1}.batch_size = 5;
[output, P] = convnet_forward(params, layers, [img1 img2 img3 img4 img5]);
[probability, predict_number] = max(P);
disp('The five prediction numbers are:')
disp(predict_number-1);