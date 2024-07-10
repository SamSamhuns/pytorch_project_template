import torch


def test_initialization(custom_model):
    """Test that custom_model initializes its parts correctly."""
    assert isinstance(
        custom_model.conv, torch.nn.Conv2d), "Convolution not initialized correctly."
    assert custom_model.conv.in_channels == 3, "Incorrect number of input channels."
    assert custom_model.conv.out_channels == 10, "Incorrect number of output filters."


def test_forward_pass_shape(custom_model):
    """Test the output shape of a forward pass."""
    input_tensor = torch.randn(
        5, 3, 64, 64)  # Batch size of 5, image size of 64x64
    output = custom_model.forward(input_tensor)
    # Adjust dimensions according to your custom_model architecture
    expected_shape = (5, 10 * 64 * 64)
    assert output.shape == expected_shape, "Output shape is incorrect."
