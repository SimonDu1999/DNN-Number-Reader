function [output] = pooling_layer_forward(input, layer)
    assert(strcmp(layer.type, 'POOLING') == 1, 'layer must be pooling layer');
    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out* w_out* c, batch_size]);
    
    
    for i=1:batch_size
        count = 1;
        img_n = reshape(input.data(:,i),[h_in*w_in,c]);       
        for j=1:c
            for m = 1:stride:w_in
                for n = 1:stride:h_in
                   % since k=2, the filter is 2*2
                   pool1 = img_n((m-1)*h_in+n,j);
                   pool2 = img_n((m-1)*h_in+n+1,j);
                   pool3 = img_n((m)*h_in+n,j);
                   pool4 = img_n((m)*h_in+n+1,j);
                   output.data(count,i) = max([pool1 pool2 pool3 pool4]);
                   count = count+1;
                end
            end
        end
    end

end

