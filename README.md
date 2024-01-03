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
