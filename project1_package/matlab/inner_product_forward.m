function [output] = inner_product_forward(input, layer, param)
assert(strcmp(layer.type, 'IP') == 1, 'layer must be inner product layer');
d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.
output.data = zeros([n, k]);

output.height = n;
output.width = 1;
output.channel = 1;
output.batch_size = input.batch_size;

% Compute the result for each batch
for i=1:input.batch_size
    input_n = gpuArray(input.data(:,i));
    output.data(:,i) = transpose(param.w) * input_n + transpose(param.b);
end

end

