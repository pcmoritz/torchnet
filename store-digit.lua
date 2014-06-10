-- get a digit from the dataset and store it into a file

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'struct'

torch.setdefaulttensortype('torch.FloatTensor')

geometry = {32,32}

nbTrainingPatches = 60000
nbTestingPatches = 10000

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

for k = 1,5 do
    img = testData[k][1][1]
    out = io.open("img-" .. tostring(k), "wb")
    for i = 1,32 do
    	for j = 1,32 do
    	    out:write(struct.pack("f", img[i][j]))
    	end	
    end
    out:close()
end