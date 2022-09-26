function showW(w, b, n)
%showW Summary of this function goes here
%   Detailed explanation goes here
z = w(n, :) + b(n);
z = abs(min(z)) + z;
image = zeros(28, 28);
for k = 1:28
    for s = 1:28
        %image(k, s) = set(index, 1 + (k - 1)*28 + s);
        image(k, s) = z((k - 1)*28 + s);
    end
end

image = image./max(image);
imshow(image, 'InitialMagnification', 'fit');
colormap(gca,cool);
end
