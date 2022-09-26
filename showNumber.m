function showNumber(set, index)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
image = zeros(28, 28);
for k = 1:28
    for s = 1:28
        image(k, s) = set(index, 1 + (k - 1)*28 + s);
    end
end
imshow(image, 'InitialMagnification', 'fit');
colormap(gca,flipud(gray));
title(num2str(set(index, 1)))
end