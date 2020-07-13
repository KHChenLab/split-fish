function codewords = NchooseKcode(n,k)
% This script is from https://www.mathworks.com/matlabcentral/answers/510687-produce-all-combinations-of-n-choose-k-in-binary
codewords = dec2bin(sum(nchoosek(2.^(0:n-1),k),2)) - '0';