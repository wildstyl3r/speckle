use tch::{
    Tensor,
    nn::{self, Module, ModuleT},
};

#[derive(Debug)]
pub struct SwiGLU {
    input: nn::Linear,
    output: nn::Linear,
    dropout: f64,
}

pub fn swiglu(path: nn::Path, emb_dim: i64, hidden_dim: i64, dropout: f64) -> SwiGLU {
    SwiGLU {
        input: nn::linear(
            &path / "glu_proj",
            emb_dim,
            2 * hidden_dim,
            Default::default(),
        ),
        output: nn::linear(&path / "glu_out", hidden_dim, emb_dim, Default::default()),
        dropout,
    }
}

impl ModuleT for SwiGLU {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let proj = self.input.forward(xs);
        let scale = proj.narrow(-1, 0, proj.size()[proj.size().len() - 1] / 2);
        let gate = proj.narrow(
            -1,
            proj.size()[proj.size().len() - 1] / 2,
            proj.size()[proj.size().len() - 1] / 2,
        );
        let gate = &gate * gate.sigmoid();
        self.output
            .forward(&(scale * gate))
            .dropout(self.dropout, train)
    }
}
