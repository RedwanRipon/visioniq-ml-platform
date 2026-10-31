import torch

from src.models.resnet import VisionIQResNet


def test_output_shape_and_class_count():
    model = VisionIQResNet(pretrained=False).eval()   # no download needed in tests
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_probabilities_sum_to_one():
    model = VisionIQResNet(pretrained=False).eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.randn(3, 3, 32, 32)), dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)