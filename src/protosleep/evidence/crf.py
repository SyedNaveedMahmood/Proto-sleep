"""Exact linear-chain CRF and a class-contrastive local/context evidence identity.

This models adjacent-stage dependence, NOT explicit duration or hour-scale physiology.
It is bidirectional offline inference. No true adjacent labels enter prediction.
"""
from __future__ import annotations

import torch
from torch import nn


class EvidenceCRF(nn.Module):
    def __init__(self, classes: int = 5, bound: float = 2.0):
        super().__init__()
        if classes < 2 or bound <= 0:
            raise ValueError("Invalid CRF config")
        self.bound = float(bound)
        self.transitions = nn.Parameter(torch.zeros(classes, classes, dtype=torch.float64))
        self.start = nn.Parameter(torch.zeros(classes, dtype=torch.float64))
        self.end = nn.Parameter(torch.zeros(classes, dtype=torch.float64))

    def potentials(self):
        return tuple(self.bound * x.tanh() for x in (self.transitions, self.start, self.end))

    def _check(self, emissions):
        if emissions.ndim != 2 or emissions.shape[0] == 0 or emissions.shape[1] != self.start.numel():
            raise ValueError("Expected a nonempty [T,C] chain")
        if not torch.isfinite(emissions).all():
            raise ValueError("Nonfinite emissions")
        return emissions.to(self.start.dtype)

    def messages(self, emissions):
        e = self._check(emissions)
        trans, start, end = self.potentials()
        alpha = [start + e[0]]
        for t in range(1, len(e)):
            alpha.append(e[t] + torch.logsumexp(alpha[-1][:, None] + trans, dim=0))
        beta = [end]
        for t in range(len(e) - 2, -1, -1):
            beta.append(torch.logsumexp(trans + e[t + 1][None, :] + beta[-1][None, :], dim=1))
        a, b = torch.stack(alpha), torch.stack(beta[::-1])
        logz = torch.logsumexp(a[-1] + end, 0)
        return a, b, logz

    def log_marginals(self, emissions):
        a, b, z = self.messages(emissions)
        return a + b - z

    def nll(self, emissions, labels):
        e = self._check(emissions)
        if labels.shape != (len(e),) or labels.dtype != torch.long:
            raise ValueError("Expected one integer label per epoch")
        if ((labels < 0) | (labels >= e.shape[1])).any():
            raise ValueError("Unknown labels must break the chain before CRF training")
        trans, start, end = self.potentials()
        score = start[labels[0]] + end[labels[-1]] + e.gather(1, labels[:, None]).sum()
        if len(e) > 1:
            score = score + trans[labels[:-1], labels[1:]].sum()
        return self.messages(e)[2] - score

    def margin_ledger(self, emissions, epoch: int, c: int, r: int):
        """log p(y_t=c|x)-log p(y_t=r|x) = local + incoming + outgoing.

        This is exact for a fixed observation sequence. Incoming/outgoing messages
        aggregate context; they are not explanations of individual neighboring waves.
        """
        e = self._check(emissions)
        if not 0 <= epoch < len(e) or not 0 <= c < e.shape[1] or not 0 <= r < e.shape[1] or c == r:
            raise ValueError("Invalid epoch/classes")
        a, b, _ = self.messages(e)
        incoming = a[epoch] - e[epoch]
        local = e[epoch, c] - e[epoch, r]
        left = incoming[c] - incoming[r]
        right = b[epoch, c] - b[epoch, r]
        return {"local": local, "left_context": left, "right_context": right,
                "margin": local + left + right}
