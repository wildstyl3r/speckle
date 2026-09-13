use tch::Tensor;

pub struct Batcher {
    data: Tensor,
    batch_size: i64,
    current_index: i64,
    total_size: i64,
    class_ix: Tensor,
    channels: i64,
    h: i64,
    w: i64,
}

impl Batcher {
    pub fn new(data: Tensor, batch_size: i64) -> Self {
        let (total_size, h, w, channels) = data.size4().unwrap();
        let class_ix = Tensor::randint(
            h * w * h * w,
            [total_size, h * w],
            (tch::Kind::Int, data.device()),
        )
        .argsort(-1, false);

        Self {
            total_size,
            class_ix,
            data,
            batch_size,
            current_index: 0,
            channels,
            h,
            w,
        }
    }
}

impl Iterator for Batcher {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.total_size {
            return None;
        }

        let end_index = std::cmp::min(self.current_index + self.batch_size, self.total_size);
        let effective_batch_size = end_index - self.current_index;
        let source_ix = self
            .class_ix
            .slice(0, self.current_index, end_index, 1)
            .view([effective_batch_size, self.h * self.w]);
        let targets = self.data.slice(0, self.current_index, end_index, 1).view([
            effective_batch_size,
            self.h,
            self.w,
            self.channels,
        ]);
        self.current_index = end_index;

        Some((source_ix, targets))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_elements = self.total_size - self.current_index;
        if remaining_elements <= 0 {
            (0, Some(0))
        } else {
            let remaining_batches =
                ((remaining_elements + self.batch_size - 1) / self.batch_size) as usize;
            (remaining_batches, Some(remaining_batches))
        }
    }
}
