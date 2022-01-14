cd '../matlab/'
layers = get_lenet();
load lenet.mat
% load the image data
im1 = imread('../images/image1.JPG');
im2 = imread('../images/image2.JPG');
im3 = imread('../images/image3.png');
im4 = imread('../images/image4.JPG');

%% classify the foreground and background pixel
im1 = rgb2gray(im1);
level1 = graythresh(im1);
BW1 = 1-imbinarize(im1,level1);
figure, imshow(BW1)

im2 = rgb2gray(im2);
level2 = graythresh(im2);
BW2 = 1-imbinarize(im2,level2);
figure, imshow(BW2)

im3 = rgb2gray(im3);
level3 = graythresh(im3);
BW3 = 1-imbinarize(im3,level3);
figure, imshow(BW3)

im4 = rgb2gray(im4);
level4 = graythresh(im4);
BW4 = 1-imbinarize(im4,level4);
figure, imshow(BW4)

%% find the bounding box
% image1
CC1 = bwconncomp(BW1);
box = regionprops(CC1, 'Area','BoundingBox');
img1_batch = zeros(28*28,CC1.NumObjects);

deletea_c = [];
% get each bouding box
for i=1:CC1.NumObjects
    if box(i).Area < 50
        deletea_c = cat(1,deletea_c,i);
        continue
    end
    BB = box(i).BoundingBox;
    xMin = ceil(BB(1));
    xMax = xMin+BB(3)-1;
    yMin = ceil(BB(2));
    yMax = yMin+BB(4)-1;
    img = BW1(yMin:yMax, xMin:xMax);
    img = padarray(img,[60,65],0);
    img = imresize(img,[28 28]);
    img1_batch(:,i) = reshape(img',[28*28,1]);
end

% delete the image that has few area which might be a CC but not a complete number
for i = 1:size(deletea_c)
    img1_batch(:,deletea_c(i)) = [];
end

% image2
CC2 = bwconncomp(BW2);
box = regionprops(CC2, 'Area', 'BoundingBox');

img2_batch = zeros(28*28,CC2.NumObjects);
deletea_c = [];
% get each bouding box
for i=1:CC2.NumObjects
    if box(i).Area < 50
        deletea_c = cat(1,deletea_c,i);
        continue
    end
    BB = box(i).BoundingBox;
    xMin = ceil(BB(1));
    xMax = xMin+BB(3)-1;
    yMin = ceil(BB(2));
    yMax = yMin+BB(4)-1;
    img = BW2(yMin:yMax, xMin:xMax);
    img = padarray(img,[50,50],0);
    img = imresize(img,[28 28]);
    img2_batch(:,i) = reshape(img',[28*28,1]);
end
for i = 1:size(deletea_c)
    img2_batch(:,deletea_c(i)) = [];
end
% image3
CC3 = bwconncomp(BW3);
box = regionprops(CC3, 'Area', 'BoundingBox');

img3_batch = zeros(28*28,CC3.NumObjects);
deletea_c = [];
% get each bouding box
for i=1:CC3.NumObjects
    if box(i).Area < 50
        deletea_c = cat(1,deletea_c,i);
        continue
    end
    BB = box(i).BoundingBox;
    xMin = ceil(BB(1));
    xMax = xMin+BB(3)-1;
    yMin = ceil(BB(2));
    yMax = yMin+BB(4)-1;
    img = BW3(yMin:yMax, xMin:xMax);
    img = padarray(img,[40,40],0);
    img = imresize(img,[28 28]);
    img3_batch(:,i) = reshape(img',[28*28,1]);
end

for i = 1:size(deletea_c)
   img3_batch(:,deletea_c(i)) = [];
end

% image4
CC4 = bwconncomp(BW4);
box = regionprops(CC4, 'Area', 'BoundingBox');

img4_batch = zeros(28*28,CC4.NumObjects);
deletea_c = [];
% get each bouding box
for i=1:CC4.NumObjects
    if box(i).Area < 50
        deletea_c = cat(1,deletea_c,i);
        continue
    end
    BB = box(i).BoundingBox;
    xMin = ceil(BB(1));
    xMax = xMin+BB(3)-1;
    yMin = ceil(BB(2));
    yMax = yMin+BB(4)-1;
    img = BW4(yMin:yMax, xMin:xMax);
    img = padarray(img,[10,10],0);
    img = imresize(img,[28 28]);
    %figure, imshow(img)
    img4_batch(:,i) = reshape(img',[28*28,1]);
end

for i = 1:size(deletea_c)
    img4_batch(:,deletea_c(i)) = [];
end

%% take each bouding box to the network

layers{1}.batch_size = size(img1_batch,2);
[output, P] = convnet_forward(params, layers, img1_batch);
[probability, predict_number] = max(P);
disp('The prediction numbers for the first image are:')
disp(predict_number-1);

layers{1}.batch_size = size(img2_batch,2);
[output, P] = convnet_forward(params, layers, img2_batch);
[probability, predict_number] = max(P);
disp('The prediction numbers for the second image are:')
disp(predict_number-1);

layers{1}.batch_size = size(img3_batch,2);
[output, P] = convnet_forward(params, layers, img3_batch);
[probability, predict_number] = max(P);
disp('The prediction numbers for the third image are:')
disp(predict_number-1);

layers{1}.batch_size = size(img4_batch,2);

[output, P] = convnet_forward(params, layers, img4_batch);
[probability, predict_number] = max(P);
disp('The prediction numbers for the fourth image are:')
disp(predict_number-1);

cd '../ec'