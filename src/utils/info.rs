use std::io::Write;

use tch::nn::VarStore;

pub fn param_count(vs: &VarStore) -> (i64, i64) {
    (
        vs.variables()
            .values()
            .map(|t| t.size().iter().product::<i64>())
            .sum(),
        vs.trainable_variables()
            .iter()
            .map(|t| t.size().iter().product::<i64>())
            .sum(),
    )
}

pub fn write_summary(vs: &VarStore, mut wr: impl Write) -> std::io::Result<()> {
    let mut sorted_vars: Vec<_> = vs.variables().into_iter().collect();
    sorted_vars.sort_by(|a, b| a.0.cmp(&b.0));
    wr.write_fmt(format_args!("{}\n", "-".repeat(85)))?;

    for (name, tensor) in sorted_vars {
        let shape = tensor.size();
        wr.write_fmt(format_args!(
            "{:<60} {:?} {}\n",
            name,
            shape,
            if tensor.requires_grad() {
                "(trainable)"
            } else {
                ""
            }
        ))?;
    }
    wr.write_fmt(format_args!("{}\n", "-".repeat(85)))?;
    Ok(())
}
