# Quant-Pack
`quant-pack` is a versatile and easy-to-extend Python package for neural 
network quantization study.

## Publications
Following original works are conducted with `quant-pack`.

Work | Publication | Module / Sub-package in `quant-pack`
-----|--------------------|-----------------------
GQ-Nets | Li et al. Rejected by [ICLR 2020](https://openreview.net/forum?id=Hkx3ElHYwS) \* | [`inverse_distillation`](quant_pack/deprecated/models/inverse_distillation)
FQN for Detection | [Li et al. CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.html) | *coming soon*

\* We decided not to put this work on arXiv until we made *enough* money with this technique.

## Reproductions
We reproduced following third-party works in `quant-pack`. Links to the 
origin publication and related code module are listed.

Work | Publication | Module / Sub-package in `quant-pack`
-----|--------------------|-----------------------
LR-Nets | [Shayer et al. ICLR 2018](https://openreview.net/forum?id=BySRH6CpW) | [`variational`](quant_pack/deprecated/models/variational)
Feature Denoising | [Xie et al. CVPR 2019](https://arxiv.org/abs/1812.03411) | [`NonLocal`](quant_pack/deprecated/models/variational/_components.py)

## License
`quant-pack` is distributed under Anti-996 license (V1.0, draft). See [LICENSE](LICENSE) for details.
