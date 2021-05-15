## Implementation of Laplacian Loss in Pytorch

Building Laplacian Pyramid using `EXPAND` operator. The `PyrUp` operator is implemented by `torch.nn.functional.conv_transpose2d`. The `PyrDown` operator is implemented by `torch.nn.functional.conv2d` using gaussian kernel.

![laplacian](./LaplacianPyramid.png)

### Comparison with OpenCV-implementation

| scale | pytorch-implementation | opencv-implementation |
| ---- | ---- | ---- |
|input| ![input](00000000.png) | ![input](00000000.png) |
|0| ![my](./torch_0.jpg) | ![offcial](./cv_0.jpg) |
|1| ![my](./torch_1.jpg) | ![offcial](./cv_1.jpg) |
|2| ![my](./torch_2.jpg) | ![offcial](./cv_2.jpg) |
|3| ![my](./torch_3.jpg) | ![offcial](./cv_3.jpg) |