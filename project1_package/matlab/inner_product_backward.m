function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));
n = input.batch_size;
diff = output.diff;
weight = param.w;
param_grad.w = 0;
param_grad.b = 0;
for i=1:n
    % use the trick to implictly perform the big Jacobian matrix
    % see the paper: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    
    param_grad.w = param_grad.w + input.data(:,i) * transpose(diff(:,i));
    param_grad.b = param_grad.b + diff(:,i)*1;
    input_od(:,i) = weight*diff(:,i);
end
param_grad.b = transpose(param_grad.b);
end
