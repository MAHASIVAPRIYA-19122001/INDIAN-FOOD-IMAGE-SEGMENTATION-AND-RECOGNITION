clc;
clear;
close all;
X=uigetfile('*.jpg;*.tiff;*.ppm;*.pgm;*.png','pick a jpge file');
I=imread(X);
im=imresize(I,[256,256]);
imshow(im);title('resized image');
I = rgb2gray(im);
figure;
imshow(I);title('gray scale image');
figure
imhist(I);title('image histogram');
figure
BW =im2bw(I,0.4);
imshow(BW);title('black and white image');
fim=mat2gray(im);
[bwfim1,level1]=fcmthresh(fim,1);
imgG=double(bwfim1).*double(rgb2gray(im));
imgClr(:,:,1)=double (bwfim1).*double(im(:,:,1));
imgClr(:,:,2)=double (bwfim1).*double(im(:,:,2));
imgClr(:,:,3)=double (bwfim1).*double(im(:,:,3));

figure;
subplot(2,2,1);
imshow(fim);title('ORIGINAL');
subplot(2,2,2);
imshow(bwfim1),title(sprintf('FCM,level=%f',level1));
subplot(2,2,3);
imshow(uint8 (imgG));title('Segmented Gray-FCM');
subplot(2,2,4);
imshow(uint8 (imgClr));title('Segmented Color-FCM');

%%FEATURES CALCULATION
%morphological operation
BWF=bwareaopen(bwfim1,3000);
img2=double(BWF).*double(rgb2gray(im));

figure;
imshow(img2,[]);title('Segmented image-After Morphology')
G=img2;

%GLCM feature extraction

g=graycomatrix(G);
stats=graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast=stats.Contrast;
Correlation=stats.Correlation;
Energy=stats.Energy;
Homogeneity=stats.Homogeneity;
Mean=mean2(G);
Standard_Deviation=std2(G);
Entropy=entropy(G);
RMS=mean2(var(double(G)));
Variance=mean2(var(double(G)));
a=sum(double(G(:)));
smoothness=1-(1/(1+a));
Kurtosis=kurtosis(double(G(:)));
skewness=skewness(double(G(:)));
%INVERSE DIFFERENCE MOMENT
m=size(G,1);
n=size(G,2);
in_diff=0;
for i= 1:m
    for j=1:n
        temp=G(i,j)./(1+(i-j).^2);
        in_diff=in_diff+temp;
    end
end
IDM=double(in_diff);
%
GLCMfeat=[Contrast,Correlation,Energy,Homogeneity,Mean,Standard_Deviation,Entropy,RMS,Variance,a,smoothness,Kurtosis,skewness];


mesg=sprintf('Successfully segmented');


msgbox(mesg)
function test_network(net,image)


I = imread(image);
R = imresize(I,[224,224]);

[Label,Probability] = classify(net,R);

figure;

imshow(R);
title({char(Label),num2str(max(Probability),6)})

end
