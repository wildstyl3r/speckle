use tch::Tensor;

pub struct Muon {
    parameters: Vec<Tensor>,
    momenta: Vec<Tensor>,

    lr: f64,
    mu: f64,
    weight_decay: f64,
    nesterov: bool,
    turbo: bool,
    eps: f64,
}

const A: f64 = 3.4445;
const B: f64 = -4.775;
const C: f64 = 2.0315;

fn newton_schulz(mut x: Tensor, turbo: bool, eps: f64) -> Tensor {
    let tall = x.size()[x.size().len() - 2] > x.size()[x.size().len() - 1];
    if tall {
        x = x.transpose(-1, -2);
    }

    let (mut a, mut x, iters) = if turbo {
        // https://arxiv.org/abs/2512.04632  Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning
        // todo: check the per-step optimized ns coefficients a,b,c
        let a = x.matmul(&x.transpose(-1, -2));
        let s = a
            .abs()
            .sum_dim_intlist(&[-1][..], false, None)
            .clamp_min(eps)
            .rsqrt();
        (
            a * s.unsqueeze(-2) * s.unsqueeze(-1),
            x * s.unsqueeze(-1),
            4,
        )
    } else {
        x /= x.frobenius_norm([-2, -1], true) + eps;
        (x.matmul(&x.transpose(-1, -2)), x, 5)
    };

    for i in 0..iters {
        if i != 0 {
            a = x.matmul_out(&a, &x.transpose(-1, -2));
        }
        let b = B * &a + C * &a.matmul(&a);
        x = (A * &x).add_out(&x, &b.matmul(&x));
    }
    if tall {
        x = x.transpose(-1, -2);
    }
    x
}

pub fn muon(
    parameters: Vec<Tensor>,
    lr: f64,
    mu: f64,
    weight_decay: f64,
    nesterov: bool,
    turbo: bool,
    eps: f64,
) -> Muon {
    Muon {
        momenta: parameters.iter().map(Tensor::zeros_like).collect(),
        parameters,
        lr,
        mu,
        weight_decay,
        nesterov,
        turbo,
        eps,
    }
}
impl Muon {
    pub fn step(&mut self) {
        tch::no_grad(|| {
            for (w, m) in self.parameters.iter_mut().zip(self.momenta.iter_mut()) {
                let _ = m.lerp_(&w.grad(), self.mu);

                let update_prepare = if self.nesterov {
                    &(w.grad().lerp_(m, self.mu))
                } else {
                    m
                };

                // https://arxiv.org/pdf/2502.16982 Muon is scalable for LLM training
                let rms_scale = 0.2
                    * (std::cmp::max(
                        update_prepare.size()[update_prepare.size().len() - 2],
                        update_prepare.size()[update_prepare.size().len() - 1],
                    ) as f64)
                        .sqrt();

                let update = newton_schulz(update_prepare.copy(), self.turbo, self.eps);
                let _ = w.subtract_(
                    &(self.lr * (rms_scale * update + w.multiply_scalar(self.weight_decay))),
                );
            }
        })
    }

    pub fn zero_grad(&mut self) {
        for p in &mut self.parameters {
            p.zero_grad();
        }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr
    }
}
