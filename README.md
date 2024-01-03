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
