import torch


def test_initialization(mnist_model):
    """Test that all layers are initialized correctly."""
    assert isinstance(
        mnist_model.conv1, torch.nn.Conv2d), "conv1 should be an instance of nn.Conv2d"
    assert isinstance(
        mnist_model.conv2, torch.nn.Conv2d), "conv2 should be an instance of nn.Conv2d"
    assert isinstance(
        mnist_model.fc1, torch.nn.Linear), "fc1 should be an instance of nn.Linear"
    assert isinstance(
        mnist_model.fc2, torch.nn.Linear), "fc2 should be an instance of nn.Linear"


def test_forward_pass_output_shape(mnist_model, device):
    """Test the output shape from the forward pass."""
    input_tensor = torch.randn(
        1, 1, 28, 28)  # Assuming input size as MNIST images
    mnist_model = mnist_model.to(device)
    output = mnist_model(input_tensor.to(device))
    assert output.shape == (
        1, 10), "Output shape should be (1, 10) for a single image."


def test_dropout_training_mode(mnist_model):
    """Ensure dropout is active during training mode."""
    mnist_model.train()  # Set mnist_model to training mode
    input_tensor = torch.randn(1, 1, 28, 28)
    output1 = mnist_model(input_tensor)
    output2 = mnist_model(input_tensor)
    assert not torch.equal(
        output1, output2), "Outputs should differ in training mode due to dropout."


def test_dropout_evaluation_mode(mnist_model):
    """Check dropout is inactive during evaluation mode."""
    mnist_model.eval()  # Set mnist_model to evaluation mode
    input_tensor = torch.randn(1, 1, 28, 28)
    output1 = mnist_model(input_tensor)
    output2 = mnist_model(input_tensor)
    assert torch.equal(
        output1, output2), "Outputs should be the same in evaluation mode as dropout is inactive."
