import torch


class Batcher:
    def __init__(self, data: torch.Tensor, batch_size: int, seed: int | None = None):
        total_size, h, w, channels = data.shape
        self.data = data
        self.batch_size = batch_size
        self.total_size = total_size
        self.channels = channels
        self.h = h
        self.w = w
        self.current_index = 0
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.generator = torch.Generator().manual_seed(seed)

    def next(self):
        if self.current_index >= self.total_size:
            return None
        end_index = min(self.current_index + self.batch_size, self.total_size)
        effective_batch_size = end_index - self.current_index
        scores = torch.randint(
            self.h * self.w * self.h * self.w,
            (effective_batch_size, self.h * self.w),
            generator=self.generator,
        )
        source_ix = scores.argsort(-1).view(effective_batch_size, self.h * self.w)
        targets = self.data[self.current_index:end_index].view(
            effective_batch_size, self.h, self.w, self.channels
        )
        if targets.dtype == torch.uint8:
            targets = targets.float().div_(255)
        self.current_index = end_index
        return source_ix, targets
