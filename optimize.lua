require 'torch'
require 'nn'
require 'struct'
require 'io'
require 'string'

torch.setdefaulttensortype('torch.FloatTensor')

model = torch.load('logs/mnist.net')
model:add(nn.LogSoftMax())

inp = io.open("digit", "rb")
data = inp:read("*all")

a = {}
i = 1
for j = 1,32*32 do
  local d
  d, i = struct.unpack("f", data, i)
  a[#a + 1] = d
end

T = torch.Tensor(a)

U = T:reshape(1,1,32,32)

outputs = model:forward(U)

print(outputs[1])

T = torch.zeros(10)
T[1] = 1.0

gradient = model:backward(U, T)

out = io.open("result", "wb")
for i = 1,32 do
    for j = 1,32 do
    	out:write(struct.pack("f", gradient[1][1][i][j]))
    end	
end
out:close()
