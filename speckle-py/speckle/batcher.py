import torch


class Batcher:
    def __init__(self, data: torch.Tensor, batch_size: int):
        total_size, h, w, channels = data.shape
        self.data = data
        self.batch_size = batch_size
        self.total_size = total_size
        self.channels = channels
        self.h = h
        self.w = w
        self.current_index = 0
        chunks = []
        chunk_size = 8192
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            chunk = torch.randint(h * w * h * w, (end - start, h * w), dtype=torch.int32).argsort(-1)
            chunks.append(chunk.to(torch.int32))
        self.class_ix = torch.cat(chunks) if chunks else torch.empty(0, h * w, dtype=torch.int32)

    def next(self):
        if self.current_index >= self.total_size:
            return None
        end_index = min(self.current_index + self.batch_size, self.total_size)
        effective_batch_size = end_index - self.current_index
        source_ix = self.class_ix[self.current_index:end_index].long().view(effective_batch_size, self.h * self.w)
        targets = self.data[self.current_index:end_index].view(
            effective_batch_size, self.h, self.w, self.channels
        )
        self.current_index = end_index
        return source_ix, targets
