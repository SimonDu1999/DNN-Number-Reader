function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
batch_size = input.batch_size;
input_od = zeros(size(input.data));
diff = output.diff;
for n = 1:batch_size
    input_n.data = input.data(:, n);
    s = size(input_n.data,1);
    for i = 1:s
        if input_n.data(i) <= 0
            der = 0;
        else
            der = 1;
        end 
        input_od(i,n) = diff(i,n)*der;
    end
end
end
