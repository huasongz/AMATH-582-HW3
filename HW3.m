clear all;clc
% load files
for i = 1:3
    for j = 1:4
        load(['cam' num2str(i) '_' num2str(j) '.mat'])
    end
end

%% Ideal Case
[a11,b11,c11,d11] = size(vidFrames1_1);
X11 = [];
Y11 = [];
for i = 1:d11
    img = rgb2gray(vidFrames1_1(:,:,:,i));
    Cam11(:,:,i) = double(img);
    img(:,1:300) = 0;
    img(:,400:end) = 0;
    img(1:200,:) = 0;
    %imshow(img)
    [Max Ind] = max(img(:));
    [y11 x11] = ind2sub(size(img),Ind);
    X11 = [X11 x11];
    Y11 = [Y11 y11];
end

[a12,b12,c12,d12] = size(vidFrames2_1);
X12 = [];
Y12 = [];
for i = 1:d12
    img = rgb2gray(vidFrames2_1(:,:,:,i));
    Cam21(:,:,i) = double(img);
    img(:,1:230) = 0;
    img(:,350:end) = 0;
    img(1:100,:) = 0;
    img(370:end,:) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y12 x12] = ind2sub(size(img),Ind);
    X12 = [X12 x12];
    Y12 = [Y12 y12];
end

[a13,b13,c13,d13] = size(vidFrames3_1);
X13 = [];
Y13 = [];
for i = 1:d13
    img = rgb2gray(vidFrames3_1(:,:,:,i));
    Cam31(:,:,i) = double(img);
    img(:,1:250) = 0;
    img(1:200,:) = 0;
    img(350:end,:) = 0;
    imshow(img)
    [Max Ind] = max(img(:));
    [y13 x13] = ind2sub(size(img),Ind);
    X13 = [X13 x13];
    Y13 = [Y13 y13];
end

figure(1)
subplot(3,2,1)
plot(X11)
ylabel('Position in x')
ylim([0 500])
title('Cam 1')
subplot(3,2,2)
plot(Y11)
title('Cam 1')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,3)
plot(X12)
ylabel('Position in x')
ylim([0 500])
title('Cam 2')
subplot(3,2,4)
plot(Y12)
title('Cam 2')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,5)
plot(X13)
ylabel('Position in x')
ylim([0 500])
title('Cam 3')
subplot(3,2,6)
plot(Y13)
title('Cam 3')
ylabel('Position in y')
ylim([0 500])
print(gcf,'-dpng','Part1_1.png');

Xmin = min([length(X11) length(X12) length(X13)]);
if length(X11) > Xmin
    X11 = X11(1:Xmin);
    Y11 = Y11(1:Xmin);
end
if length(X12) > Xmin
    X12 = X12(1:Xmin);
    Y12 = Y12(1:Xmin);
end
if length(X13) > Xmin
    X13 = X13(1:Xmin);
    Y13 = Y13(1:Xmin);
end

Coord = [X11;Y11;X12;Y12;X13;Y13];
[m,n]=size(Coord); % compute data size
mn=mean(Coord,2); % compute mean for each row
Coord=Coord-repmat(mn,1,n); % subtract mean

Cx=(1/(n-1))*Coord*Coord'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
V=V(:,m_arrange);
Y=V'*Coord; % produce the principal components projection

figure(2)
plot(Y(1,:))
xlabel('Frame number')
ylabel('Position in x')
title('Ideal case')
print(gcf,'-dpng','Part1_2.png');

%% noisy case
[a21,b21,c21,d21] = size(vidFrames1_2);
X21 = [];
Y21 = [];
for i = 1:d21
    img = rgb2gray(vidFrames1_2(:,:,:,i));
    Cam12(:,:,i) = double(img);
    img(:,1:300) = 0;
    img(:,400:end) = 0;
    img(1:200,:) = 0;
    %imshow(img)
    [Max Ind] = max(img(:));
    [y21 x21] = ind2sub(size(img),Ind);
    X21 = [X21 x21];
    Y21 = [Y21 y21];
end

[a22,b22,c22,d22] = size(vidFrames2_2);
X22 = [];
Y22 = [];
for i = 1:d22
    img = rgb2gray(vidFrames2_2(:,:,:,i));
    Cam22(:,:,i) = double(img);
    img(:,1:200) = 0;
    img(:,400:end) = 0;
    %imshow(img)
    [Max Ind] = max(img(:));
    [y22 x22] = ind2sub(size(img),Ind);
    X22 = [X22 x22];
    Y22 = [Y22 y22];
end

[a23,b23,c23,d23] = size(vidFrames3_2);
X23 = [];
Y23 = [];
for i = 1:d23
    img = rgb2gray(vidFrames3_2(:,:,:,i));
    Cam32(:,:,i) = double(img);
    img(:,1:250) = 0;
    img(1:200,:) = 0;
    img(350:end,:) = 0;
    %imshow(img)
    [Max Ind] = max(img(:));
    [y23 x23] = ind2sub(size(img),Ind);
    X23 = [X23 x23];
    Y23 = [Y23 y23];
end

Xmin2 = min([length(X21) length(X22) length(X23)]);
if length(X21) > Xmin2
    X21 = X21(1:Xmin2);
    Y21 = Y21(1:Xmin2);
end
if length(X22) > Xmin2
    X22 = X22(1:Xmin2);
    Y22 = Y22(1:Xmin2);
end
if length(X23) > Xmin2
    X23 = X23(1:Xmin2);
    Y23 = Y23(1:Xmin2);
end

figure(3)
subplot(3,2,1)
plot(X21)
ylabel('Position in x')
ylim([0 500])
title('Cam 1')
subplot(3,2,2)
plot(Y21)
title('Cam 1')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,3)
plot(X22)
ylabel('Position in x')
ylim([0 500])
title('Cam 2')
subplot(3,2,4)
plot(Y22)
title('Cam 2')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,5)
plot(X23)
ylabel('Position in x')
ylim([0 500])
title('Cam 3')
subplot(3,2,6)
plot(Y23)
title('Cam 3')
ylabel('Position in y')
ylim([0 500])
print(gcf,'-dpng','Part2_1.png');


Coord2 = [X21;Y21;X22;Y22;X23;Y23];
[m,n]=size(Coord2); % compute data size
mn=mean(Coord2,2); % compute mean for each row
Coord2=Coord2-repmat(mn,1,n); % subtract mean

Cx=(1/(n-1))*Coord2*Coord2'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
V=V(:,m_arrange);
Y2=V'*Coord2; % produce the principal components projection

figure(4)
plot(Y2(1,:))
xlabel('Frame number')
ylabel('Position in x')
title('noisy case')
print(gcf,'-dpng','Part2_2.png');
%% Horizontal Displacement
[a31,b31,c31,d31] = size(vidFrames1_3);
X31 = [];
Y31 = [];
for i = 1:d31
    img = rgb2gray(vidFrames1_3(:,:,:,i));
    Cam13(:,:,i) = double(img);
    img(:,1:250) = 0;
    img(:,400:end) = 0;
    img(1:200,:) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y31 x31] = ind2sub(size(img),Ind);
    X31 = [X31 x31];
    Y31 = [Y31 y31];
end

[a32,b32,c32,d32] = size(vidFrames2_3);
X32 = [];
Y32 = [];
for i = 1:d32
    img = rgb2gray(vidFrames2_3(:,:,:,i));
    Cam23(:,:,i) = double(img);
    img(:,1:200) = 0;
    img(:,400:end) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y32 x32] = ind2sub(size(img),Ind);
    X32 = [X32 x32];
    Y32 = [Y32 y32];
end

[a33,b33,c33,d33] = size(vidFrames3_3);
X33 = [];
Y33 = [];
for i = 1:d33
    img = rgb2gray(vidFrames3_3(:,:,:,i));
    Cam33(:,:,i) = double(img);
    img(:,1:250) = 0;
    img(1:200,:) = 0;
    img(350:end,:) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y33 x33] = ind2sub(size(img),Ind);
    X33 = [X33 x33];
    Y33 = [Y33 y33];
end

Xmin3 = min([length(X31) length(X32) length(X33)]);
if length(X31) > Xmin3
    X31 = X31(1:Xmin3);
    Y31 = Y31(1:Xmin3);
end
if length(X32) > Xmin3
    X32 = X32(1:Xmin3);
    Y32 = Y32(1:Xmin3);
end
if length(X33) > Xmin3
    X33 = X33(1:Xmin3);
    Y33 = Y33(1:Xmin3);
end

figure(5)
subplot(3,2,1)
plot(X31)
ylabel('Position in x')
ylim([0 500])
title('Cam 1')
subplot(3,2,2)
plot(Y31)
title('Cam 1')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,3)
plot(X32)
ylabel('Position in x')
ylim([0 500])
title('Cam 2')
subplot(3,2,4)
plot(Y32)
title('Cam 2')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,5)
plot(X33)
ylabel('Position in x')
ylim([0 500])
title('Cam 3')
subplot(3,2,6)
plot(Y33)
title('Cam 3')
ylabel('Position in y')
ylim([0 500])
print(gcf,'-dpng','Part3_1.png');


Coord3 = [X31;Y31;X32;Y32;X33;Y33];
[m,n]=size(Coord3); % compute data size
mn=mean(Coord3,2); % compute mean for each row
Coord3=Coord3-repmat(mn,1,n); % subtract mean

Cx=(1/(n-1))*Coord3*Coord3'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
V=V(:,m_arrange);
Y13=V'*Coord3; % produce the principal components projection

figure(6)
plot(Y13(1,:))
xlabel('Frame number')
ylabel('Position in x')
title('horizontal displacement')
print(gcf,'-dpng','Part3_2.png');
%% horizontal displacement and rotation:
[a41,b41,c41,d41] = size(vidFrames1_4);
X41 = [];
Y41 = [];
for i = 1:d41
    img = rgb2gray(vidFrames1_4(:,:,:,i));
    Cam14(:,:,i) = double(img);
    img(:,1:300) = 0;
    img(:,450:end) = 0;
    img(1:200,:) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y41 x41] = ind2sub(size(img),Ind);
    X41 = [X41 x41];
    Y41 = [Y41 y41];
end

[a42,b42,c42,d42] = size(vidFrames2_4);
X42 = [];
Y42 = [];
for i = 1:d42
    img = rgb2gray(vidFrames2_4(:,:,:,i));
    Cam24(:,:,i) = double(img);
    img(:,1:200) = 0;
    img(:,450:end) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y42 x42] = ind2sub(size(img),Ind);
    X42 = [X42 x42];
    Y42 = [Y42 y42];
end

[a34,b34,c34,d34] = size(vidFrames3_4);
X43 = [];
Y43 = [];
for i = 1:d34
    img = rgb2gray(vidFrames3_4(:,:,:,i));
    Cam34(:,:,i) = double(img);
    img(:,1:250) = 0;
    img(1:100,:) = 0;
    img(300:end,:) = 0;
    % imshow(img)
    [Max Ind] = max(img(:));
    [y43 x43] = ind2sub(size(img),Ind);
    X43 = [X43 x43];
    Y43 = [Y43 y43];
end

Xmin4 = min([length(X41) length(X42) length(X43)]);
if length(X41) > Xmin4
    X41 = X41(1:Xmin4);
    Y41 = Y41(1:Xmin4);
end
if length(X42) > Xmin4
    X42 = X42(1:Xmin4);
    Y42 = Y42(1:Xmin4);
end
if length(X43) > Xmin4
    X43 = X43(1:Xmin4);
    Y43 = Y43(1:Xmin4);
end


figure(7)
subplot(3,2,1)
plot(X41)
ylabel('Position in x')
ylim([0 500])
title('Cam 1')
subplot(3,2,2)
plot(Y41)
title('Cam 1')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,3)
plot(X42)
ylabel('Position in x')
ylim([0 500])
title('Cam 2')
subplot(3,2,4)
plot(Y42)
title('Cam 2')
ylabel('Position in y')
ylim([0 500])
subplot(3,2,5)
plot(X43)
ylabel('Position in x')
ylim([0 500])
title('Cam 3')
subplot(3,2,6)
plot(Y43)
title('Cam 3')
ylabel('Position in y')
ylim([0 500])
print(gcf,'-dpng','Part4_1.png');


Coord4 = [X41;Y41;X42;Y42;X43;Y43];
[m,n]=size(Coord4); % compute data size
mn=mean(Coord4,2); % compute mean for each row
Coord4=Coord4-repmat(mn,1,n); % subtract mean

Cx=(1/(n-1))*Coord4*Coord4'; % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
V=V(:,m_arrange);
Y4=V'*Coord4; % produce the principal components projection

figure(8)
plot(Y4(1,:))
xlabel('Frame number')
ylabel('Position in x')
title('horizontal displacement and rotation')
print(gcf,'-dpng','Part4_2.png');
