function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices.
% Total number of words in the dictionary
n = 1899;

len=length(word_indices);
for i=1:len
x(word_indices(i))=1;
end
end
