# Studying Gaussian Splatting Source Code
This article aims to study and share "3D Gaussian Splatting for Real-Time Radiance Field Rendering" source code.

From 'https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py'

1. This code defines a GaussianModel class used to represent a Gaussian distribution model.
   
2. The GaussianModel class contains multiple attributes and methods:

- _xyz: Stores 3D point cloud coordinates.
- _features_dc: Stores the features of DC components.
- _features_rest: Stores the features of non-DC components.
- _scaling: Stores scaling information.
- _rotation: Stores rotation information.
- _opacity: Stores opacity information.

- setup_functions(): Sets up some internal functions.
- capture(): Returns the state information of the model.
- restore(): Restores the model from a saved state.
- get_xxx(): Methods to retrieve various attribute information.
- oneupSHdegree(): Increases the spherical harmonics (SH) degree.
- create_from_pcd(): Creates a model from a point cloud.
- training_setup(): Sets up model training.
- update_learning_rate(): Updates the learning rate.
- construct_list_of_attributes(): Constructs a list of attributes.
- save_ply(): Saves the model as a ply file.
- reset_opacity(): Resets opacity.
- load_ply(): Loads the model from a ply file.
- replace_tensor_to_optimizer(): Replaces tensors that need optimization.
- prune_points(): Prunes the point cloud.
- densify_and_split(): Densifies and splits the point cloud.
- densify_and_clone(): Densifies and clones the point cloud.
- densify_and_prune(): Densifies and prunes the point cloud.
- add_densification_stats(): Adds densification statistics.

3. This model is mainly used to represent Gaussian distribution point clouds, including coordinates, features, transformations, and more. It can load point clouds for modeling, perform operations like learning rate adjustment, pruning, densification, and is a trainable and optimizable Gaussian model.

`create_from_pcd()`

The method `create_from_pcd()` is an important method in the GaussianModel class, and it is used to create a GaussianModel object from point cloud data.

The main steps of this method are as follows:

1. Convert the input point cloud into a Tensor and store it in `fused_point_cloud`.

2. Calculate the RGB colors of the point cloud and convert them into a spherical harmonics (SH) representation, which is then stored in `fused_color`.

3. Initialize a feature tensor `features` filled with zeros and store the DC components from `fused_color` in `features[:,:,0:1]`.

4. Compute the squared Euclidean distances between points as an initial scale parameter, with a shape of `[number of points, 3]`.

5. Initialize rotation as a quaternion filled with ones, indicating no rotation, with a shape of `[number of points, 4]`.

6. Initialize opacity as 0.1, with a shape of `[number of points, 1]`.

7. Convert coordinates, features, scaling, rotation, and opacity into Parameters with `require_grad=True` to make them amenable to training optimization.

8. Print the number of points and complete the initialization of the GaussianModel object.

This step is crucial for transforming the original point cloud into an optimizable Gaussian model. It initializes various parameters such as point cloud features and transformation parameters and converts them into Parameters with gradients enabled, laying the foundation for subsequent model optimization and training.

In the `create_from_pcd()` method, the coordinates, features, scaling, rotation, and opacity are all initialized as trainable parameters.

The specific implementation in the code is as follows:

```python
self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) 

self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))

self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

self._scaling = nn.Parameter(scales.requires_grad_(True))

self._rotation = nn.Parameter(rots.requires_grad_(True))

self._opacity = nn.Parameter(opacities.requires_grad_(True))
```

Here, coordinates, features, scaling, rotation, and opacity are wrapped in nn.Parameter objects with requires_grad=True set. This means that these variables will become model parameters, and during the training process, the computation graph will automatically track the gradients of these parameters for backpropagation and optimization.

Setting these attributes as trainable parameters is a crucial step in building an optimizable Gaussian model. Subsequently, you can update these parameters by adjusting the learning rate, choosing an optimizer, and using optimization techniques, allowing the model to gradually approximate the target point cloud distribution.

In summary, this indeed converts the coordinates and attributes of the point cloud into trainable parameters, preparing the model for optimization.

In the `create_from_pcd()` method, the point cloud coordinates `xyz` and RGB colors are extracted from `BasicPointCloud`, and the colors are transformed into features represented in spherical harmonics (SH) form:

```
fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
features[:, :3, 0 ] = fused_color
```

The SH features consist of two parts: `_features_dc` and `features_rest`:

- `_features_dc` contains the low-frequency components corrresponding to the first 3 SH coefficients.

- `_features_rest` contains the high-frequency SH coefficients beyond the low-frequency components.

So, it can be observed that this model uses the SH transform of the point cloud as a feature representation and processes it into low-frequency and high-frequency components. SH transformation is well-suited for representing both color and geometric features of point clouds.

`SH`

Spherical harmonics (SH) are a set of basics functions defined on the surface of a sphere, often used to represent attributes of 3D scenes such as the color and normals of point clouds.

The main characteristics of SH include:

- SH is a set of orthogonal basis functions on the spherical surface that can represent any function defined on the sphere.

- By linearly combining SH basis functions, one can approximate spherical functions.

- SH basis functions are associated with spherical coordinate systems and can represent the intensity of a signal in different directions.

- SH exhibits rotation invariance, meaning that the representation remains the same under coordinate system rotations.

- Low-order SH corresponds to low-frequency signals, while high-order SH corresponds to high-frequency details.

- Attributes of 3D point clouds, such as color and normals, can be represented using SH, where low-order SH represents global properties and high-order SH captures details.

The primary steps to convert RGB colors of point clouds into SH representation are as follows:

1. Transform RGB colors into a new coordinate system, such as world coordinates or local coordinates.

2. Convert colors from Cartesian coordinates to spherical coordinates.

3. Based on the direction of points, calculate the intensities corresponding to different orders of SH basis functions as new features.

4. Low-order SH features represent the overall color distribution, while high-order features capture finer details.

In summary, SH provides a method to represent point cloud attributes using spherical harmonic basis functions, allowing the representation of both global and local features of attributes such as color and normals. Therefore, it is well-suited as a feature descriptor for point clouds.

From 'https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/gaussian_renderer/__init__.py'

This code implements the rendering process for a Gaussian model:

1. Create a zero tensor `screenspace_points` to obtain gradients for 2D means.

2. Set up rasterization configurations, including image size, view frustum parameters, and more.

3. Create a `GaussianRasterizer` for rasterization.

4. Provide parameters for the 3D Gaussian model, such as mean, variance, features, etc. You can also precompute covariance and color on the Python side to accelerate rendering.

5. The rasterization process will output the rendered image and the radius of Gaussians in screen space.

6. Filter out invisible Gaussians based on the radius.

7. Return the rendering result, screen space points, visibility filtering, and radius information.

This code effectively implements the rendering process from a 3D Gaussian model to a 2D image and computes auxiliary information for subsequent model optimization.

The rendering process from a 3D Gaussian model to a 2D image can be summarized as follows:

1. Calculate the 2D coordinates (means2D) on the image plane where each 3D Gaussian projects under the current viewpoint.

2. Based on the 3D variance-covariance matrix of each Gaussian, calculate the 2D variance-covariance matrix after projection onto the image plane.

3. Utilize 2D means and 2D variance-covariance matrices to draw the contours of each Gaussian on the image.

4. Combine all the Gaussians to compose the entire scene image.

In more detail:

- The 3D coordinates (means3D) of each point are transformed using the model-view projection matrix to obtain screenspace_points, which are the 2D means (means2D).

- If the 3D covariance matrix cov3D_precomp is precomputed, it is used directly; otherwise, it is inferred based on scaling and rotation information.

- The `GaussianRasterizer` applies Gaussian filtering based on means2D and the variance-covariance matrices, resulting in the rendered image (rendered_image).

- Additionally, the radii of each Gaussian on the screen are provided to determine visibility.

The core idea is to leverage the projection properties of Gaussian distributions to project 3D Gaussians onto a 2D plane through transformation, enabling image synthesis.

In three-dimensional space, an iso-surface (iso-probability density surface) based on a Gaussian distribution will be an ellipsoid, not a two-dimensional ellipse. The center of this ellipsoid will be located at the mean of the three-dimensional Gaussian distribution, while its shape, size, and orientation will be determined by the covariance matrix.

In three-dimensional space, the covariance matrix has three eigenvalues and corresponding eigenvectors. These eigenvalues and eigenvectors represent the lengths and directions of the three principal axes of the ellipsoid. Larger eigenvalues correspond to longer extensions along the respective axes, and the directions of the eigenvectors determine the orientation of the ellipsoid in space.

```python
# Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
```

This code's purpose is to convert (rasterize) three-dimensional objects, possibly ellipsoids, represented as Gaussian distributions into two-dimensional images. This process involves projecting from three-dimensional space to the two-dimensional image space, as well as potentially handling shading and opacity. Here's a potential interpretation of each parameter:

1. **means3D**: These are the means of the three-dimensional Gaussian distributions, representing the center positions of each Gaussian distribution (or ellipsoid) in three-dimensional space.

2. **means2D**: These might be the means of the three-dimensional Gaussian distributions projected onto the two-dimensional image plane.

3. **shs**: These could represent Spherical Harmonics coefficients, which might be used to describe lighting or color characteristics for each Gaussian distribution.

4. **colors_precomp**: These could be precomputed color values used for shading each Gaussian distribution on the image.

5. **opacities**: These are the opacities of each Gaussian distribution, possibly used to control their visibility in the final image.

6. **scales**: These are scale factors used to adjust the size of each Gaussian distribution.

7. **rotations**: These represent the rotations of each Gaussian distribution, possibly in the form of quaternions, rotation matrices, or Euler angle representations.

8. **cov3D_precomp**: This is the precomputed covariance matrix of the three-dimensional Gaussian distributions, determining the shape and orientation of each ellipsoid.

The function returns two values:

- **rendered_image**: This is the rasterized two-dimensional image, containing all visible Gaussian distributions.
- **radii**: These might be the radii or sizes of each Gaussian distribution on the image, potentially used for further processing or analysis.

From 'https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py'

- This code implements the neural network training process for Gaussian body rendering.

- It defines two main functions: `training()` for the training loop and `prepare_output_and_logger()` for initializing the output directory and logging.

- The `training()` function includes key training loop steps such as dataset loading, rendering, loss computation, and parameter updates.

- The rendering process involves calling the `render()` function, which generates images based on camera parameters, Gaussian body parameters, and rendering settings.

- The loss calculation includes both L1 loss and SSIM loss components.

- During training, the Gaussian body's order is gradually increased, and sparseification of the Gaussian body is performed after a certain number of iterations.

- Periodic evaluations on the test dataset are conducted during training, and the training process is visualized using TensorBoard.

- After training, the Gaussian body parameters are saved.

In summary, this is a training script for a Gaussian body rendering neural network that encompasses complete functionalities for training, testing, visualization, and model saving.

```Cite
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
