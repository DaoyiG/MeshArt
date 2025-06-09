import torch


def softargmax(logits):
    smax = torch.nn.functional.softmax(logits, dim=-1)  # N x 9 x 257
    tokens = torch.tensor(list(range(logits.shape[-1])), dtype=torch.float32, device=logits.device).reshape(1, 1, -1).expand_as(smax)
    return torch.sum(tokens * smax, dim=-1)


if __name__ == '__main__':
    logits_ = torch.tensor([
        [  # 2 x 3 x 4
            [
                [2, 0.1, 0.2, 0.5]
            ],
            [
                [0.7, 0.1, 0.2, 5]
            ],
            [
                [0, 10, 0.4, 0.5]
            ],
        ],
        [  # 2 x 3 x 4
            [
                [0, 0.1, 0.2, 0.9]
            ],
            [
                [0.7, 0.1, 100, 5]
            ],
            [
                [0, 1, 4, 5]
            ],
        ]
    ], requires_grad=True)
    print(softargmax(logits_), softargmax(logits_).requires_grad)
