%create 10 figures each of digits from mfeat_pix, yielding the figure from
%the lecture notes

% insert your local path where you run your digits investigations from. The
% SOM toolbox should be in a subdirectory of your working directory (which is here:
% DigitsML)

% add all subdirectories to the Matlab search paths
addpath(genpath('.'));

% load the pixel data, resulting in a matlab matrix of dim 2000 x 240
% called "mfeat_pix"
load mfeat-pix.txt -ascii;

% plot the figure from the lecture notes. 
figure(1);
for i = 1:10
    for j = 1:10
        pic = mfeat_pix(200 * (i-1)+j ,:);  
        picmatreverse = zeros(15,16);
        % the filling of (:) is done columnwise!
        picmatreverse(:)= - pic;
        picmat = zeros(15,16);
        for k = 1:15
            picmat(:,k)=picmatreverse(:,16-k);
        end
        subplot(10,10,(i-1)* 10 + j);
        pcolor(picmat');
        axis off;
        colormap(gray(10));
    end
end

