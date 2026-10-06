import torch

from torch import nn

from minimind_lab.models.embedding import TokenEmbedding

def test_embedding_forward_matches_torch():
    manual_embedding = TokenEmbedding(10, 3)
    torch_embedding = nn.Embedding(10, 3)

    with torch.no_grad():
        manual_embedding.weight.copy_(torch_embedding.weight)

    input_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])

    manual_output = manual_embedding(input_ids)
    torch_output = torch_embedding(input_ids)

    assert manual_output.shape == (2, 3, 3)
    assert torch_output.shape == (2, 3, 3)
    torch.testing.assert_close(
        manual_output, 
        torch_output
        )


def test_embedding_gradients_match_torch():
    manual_embedding = TokenEmbedding(10, 3)
    torch_embedding = nn.Embedding(10, 3)

    with torch.no_grad():
        manual_embedding.weight.copy_(torch_embedding.weight)

    input_ids = torch.tensor([[1, 2, 1], [3, 0, 2]])

    manual_output = manual_embedding(input_ids)
    torch_output = torch_embedding(input_ids)

    manual_output.sum().backward()
    torch_output.sum().backward()

    expected_grad = torch.tensor([
        [1.0 ,1.0 ,1.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    assert torch.equal(manual_embedding.weight.grad, torch_embedding.weight.grad)
    torch.testing.assert_close(
        manual_embedding.weight.grad, 
        expected_grad)







