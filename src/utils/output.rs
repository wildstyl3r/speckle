use tch::{Kind, Tensor};

pub fn patch_coords(centers: &Tensor, p_side: i64, width: i64) -> Tensor {
    let (b, n, _) = centers.size3().unwrap();
    let padding = p_side / 2;
    let padded_w = width + 2 * padding;
    let steps =
        Tensor::arange(p_side, (centers.kind(), centers.device())).to_device(centers.device());
    let dy = steps.view([-1, 1]).expand([p_side, p_side], false);
    let dx = steps.view([1, -1]).expand([p_side, p_side], false);

    let absolute_coords = centers.view([b, n, 1, 1, 2])
        + Tensor::stack(&[dy, dx], -1).view([1, 1, p_side, p_side, 2]);
    //[b,n,p*p]
    (absolute_coords.select(-1, 0) * padded_w + absolute_coords.select(-1, 1))
        .view([b, n * p_side * p_side])
}

pub fn make_pictures(
    flat_coords: &Tensor, //[B,N*p*p]
    patches: &Tensor,     //[B,N,P*P,C]
    scores: &Tensor,      //[B,N,P*P]
    p_side: i64,
    hwc: (i64, i64, i64),
) -> (Tensor, Tensor, Tensor) {
    let (height, width, channels) = hwc;
    let (b, npp) = flat_coords.size2().unwrap();
    let padding = p_side / 2;
    let padded_h = height + 2 * padding;
    let padded_w = width + 2 * padding;
    let padded_hw = padded_h * padded_w;

    let pixel_max = Tensor::full(
        [b, padded_hw],
        f64::NEG_INFINITY,
        (Kind::Float, scores.device()),
    )
    .internal_scatter_reduce(1, flat_coords, &scores.view([b, npp]), "amax", true);
    //[b,n,p*p]
    let exp_scores = (&scores.view([b, npp]) - &pixel_max.gather(1, flat_coords, false))
        .exp()
        .view([b, npp]); //batch_max).exp().view([b, npp]);

    //[b,n*p*p,c]
    let weighted_values = patches.reshape([b, npp, channels]) * &exp_scores.unsqueeze(-1);

    let numerator = Tensor::zeros([b, padded_hw, channels], (Kind::Float, patches.device()))
        .scatter_add_(
            1,
            &flat_coords.unsqueeze(-1).expand([b, npp, channels], false),
            &weighted_values,
        );

    let denominator = Tensor::zeros([b, padded_hw], (Kind::Float, patches.device())).scatter_add_(
        1,
        flat_coords,
        &exp_scores,
    );

    let mask_has_data = Tensor::full([b, padded_hw], 0, (Kind::Int64, patches.device()))
        .scatter_value_(-1, flat_coords, 1)
        .eq(1);
    (
        (&numerator
            / denominator
                .where_self(
                    &mask_has_data,
                    &Tensor::ones([], (Kind::Float, patches.device())),
                )
                .view([b, padded_hw, 1]))
        .view([b, padded_h, padded_w, channels])
        .narrow(1, padding, height)
        .narrow(2, padding, width),
        (pixel_max.unsqueeze(-1) / denominator.view([b, padded_hw, 1]))
            .view([b, padded_h, padded_w, 1])
            .narrow(1, padding, height)
            .narrow(2, padding, width),
        mask_has_data
            .view([b, padded_h, padded_w])
            .narrow(1, padding, height)
            .narrow(2, padding, width),
    )
}
