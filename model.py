import torch


class Baseline(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        channels=[128, 128, 128],
        kernel_sizes=[5, 3, 3],
        dilations=[1, 2, 2],
        num_classes=1_000,
    ):
        super().__init__()
        raise NotImplementedError

    def make_block(self, in_channels, out_channels, kernel_size, dilation):
        raise NotImplementedError

    def average_pooling(self, x, lens):
        raise NotImplementedError

    def forward(self, padded_data):
        x = padded_data.data
        lens = padded_data.lengths
        x = self.backbone(x)
        x = self.average_pooling(x, lens)
        x = self.classifier(x)
        return x


MODELS = {
    "baseline": Baseline,
}


def setup_model(*, model_name, **kwargs):
    return MODELS[model_name](**kwargs)
