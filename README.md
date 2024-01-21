# WTSR
Lightweight Wavelet-Based Transformer for Image Super-Resolution

This code repository includes the source code for the [Lightweight Wavelet-Based Transformer for Image Super-Resolution](https://link.springer.com/chapter/10.1007/978-3-031-20868-3_27):

![WTSR](./imgs/flowchart.eps "The flowchart of WTSR")Suffering from the inefficiency of deeper and wider networks, most remarkable super-resolution algorithms cannot be easily applied to real-world scenarios, especially resource-constrained devices. In this paper, to concentrate on fewer parameters and faster inference, an end-to-end Wavelet-based Transformer for Image Super-resolution (WTSR) is proposed. Different from the existing approaches that directly map low-resolution (LR) images to high-resolution (HR) images, WTSR also implicitly mines the self-similarity of image patches by a lightweight Transformer on the wavelet domain, so as to balance the model performance and computational cost. More specifically, a two-dimensional stationary wavelet transform is designed for the mutual transformation between feature maps and wavelet coefficients, which reduces the difficulty of mining self-similarity. For the wavelet coefficients, a Lightweight Transformer Backbone (LTB) and a Wavelet Coefficient Enhancement Backbone (WECB) are proposed to capture and model the long-term dependency between image patches. Furthermore, a Similarity Matching Block (SMB) is investigated to combine global self-similarity and local self-similarity in LTB. Experimental results show that our proposed approach can achieve better super-resolution performance on the multiple public benchmarks with less computational complexity.

# Training
1. Download training set DIV2K [[Official Link]](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and Flickr2K [[Official Link]](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)
2. Run `./scripts/Prepare_TrainData_HR_LR.m` in Matlab to generate HR/LR training pairs with corresponding degradation model and scale factor.
3. Edit `./options/train/train_SRFBN_example.json` for your needs.
4. Then, run command:
   ```
   python train.py -opt options/train/DTM_x4.json
   ```
5. You can monitor the training process in `./experiments`.
6. Edit `./options/train/train_SRFBN_example.json` for coresponding train config.
7. Finally, run command:
   ```
   python test.py -opt options/train/DTM_x4.json
   ```
   for test.

# Citation
If you find our work useful in your research, please cite the following paper:
```
@inproceedings{ran2022lightweight,
  title={Lightweight wavelet-based transformer for image super-resolution},
  author={Ran, Jinye and Zhang, Zili},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={368--382},
  year={2022},
  organization={Springer}
}
```
# Acknowledgements
Our code structure is derived from [SRFBN](https://github.com/Paper99/SRFBN_CVPR19).
