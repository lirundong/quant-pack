# Quant-Pack

`quant-pack` is a versatile and easy-to-extend Python package for neural 
network quantization study.

## Publications
Following original works are conducted with `quant-pack`.

Work | Publication | Module in `quant-pack`
-----|--------------------|-----------------------
GQ-Nets | *Reveal after reviewing* | [`inverse_distillation`](quant_pack/modeling/inverse_distillation)

## Reproductions
Following third-party works are reproduced in `quant-pack`. Links to the 
origin publication and related code module are listed.

Work | Publication | Module in `quant-pack`
-----|--------------------|-----------------------
LR-Nets | [ICLR 2018](https://openreview.net/forum?id=BySRH6CpW) | [`variational`](quant_pack/modeling/variational)
Feature Denoising | [CVPR 2019](https://arxiv.org/abs/1812.03411) | [`NonLocal`](quant_pack/modeling/variational/_components.py)

## License
`quant-prob` is distributed under Anti-996 license (V1.0, draft). See [LICENSE](LICENSE) for details.
