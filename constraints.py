import torch
import torch.nn as nn


__all__ = ['WeightsConstraints']


class WeightsConstraints(nn.Module):

    def forward(self,
                current_model: nn.Module,
                reference_model: nn.Module,
                decay: float,
                p: int) -> torch.Tensor:
        """
        Implements weight decay over the current model to be close to the
        weights of a reference model.
        make sure both models are on the same device.
        Args:
            current_model: current model being trained.
            reference_model: reference model (no grad). fixed model.
            decay: float. lambda.
            p: int. the p for Lp norm.  {1, 2}.

        Returns: torch.Tensor: the scalar loss.
        """

        assert p in [1, 2]

        loss = torch.tensor([0.0], dtype=torch.float, requires_grad=True,
                            device=next(current_model.parameters()).device)

        for pc, pr in zip(current_model.parameters(),
                          reference_model.parameters()):

            assert pc.shape == pr.shape

            if p == 1:
                norm = (pc - pr).abs().sum()
            elif p == 2:
                norm = ((pc - pr) ** 2).sum()
            else:
                raise NotImplementedError

            loss = loss + norm

        return decay * loss




