import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, sub_module, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.sub_module = sub_module
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.sub_module._modules.items():
            # if name is "classifier":
            #     x = x.reshape(x.size(0), -1)
            if name is "base":
                for block_name, cnn_block in module._modules.items():
                    x = cnn_block(x)
                    if block_name in self.extracted_layers:
                        outputs.append(x)
        return outputs[0]