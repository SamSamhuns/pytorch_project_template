import torch


def test_forward_pass_shape(custom_model):
    """Test the output shape of a forward pass."""
    input_tensor = torch.randn(
        5, 3, 64, 64)  # Batch size of 5, image size of 64x64
    output = custom_model.forward(input_tensor)
    # Adjust dimensions according to your custom_model architecture
    expected_shape = (5, 10)
    assert output.shape == expected_shape, "Output shape is incorrect."
