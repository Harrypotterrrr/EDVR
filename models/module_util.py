import tensorflow as tf
from tensorflow import keras


class ResidualBlock_noBN(tf.keras.layers.Layer):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()

        self.conv1 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same",
                            kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same",
                            kernel_initializer=tf.keras.initializers.HeNormal())
        self.relu = keras.layers.ReLU()

    def __call__(self, x):
        identify = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identify + out

class Module(tf.keras.Model):
    def __init__(self, block, n_layers):
        super(Module, self).__init__()
        self.n_layers = n_layers
        self.module_list = tf.keras.Sequential()
        for i in range(self.n_layers):
            self.module_list.add(block())
    def __call__(self, x):
        out = self.module_list(x)
        return out

def make_layer(block, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append(block())
    return keras.Sequential(layers)


def residualblock_nobn(x, nf=64):
    identity = x
    out = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")(x)
    out = tf.nn.relu(out)
    out = keras.layers.Conv2D(nf, (3, 3), (1, 1), "same")(out)
    return identity + out


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
  """Similar to Matlab's interp2 function.

  Finds values for query points on a grid using bilinear interpolation.

  Args:
    grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
    query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    name: a name for the operation (optional).
    indexing: whether the query points are specified as row and column (ij),
      or Cartesian coordinates (xy).

  Returns:
    values: a 3-D `Tensor` with shape `[batch, N, channels]`

  Raises:
    ValueError: if the indexing mode is invalid, or if the shape of the inputs
      invalid.
  """
  if indexing != 'ij' and indexing != 'xy':
    raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

  with tf.name_scope(name):
    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)
    shape = tf.shape(grid)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    query_type = query_points.dtype
    grid_type = grid.dtype
    num_queries = tf.shape(query_points)[1]


    alphas = []
    floors = []
    ceils = []

    index_order = [0, 1] if indexing == 'ij' else [1, 0]
    unstacked_query_points = tf.unstack(query_points, axis=2)

    for dim in index_order:
      with tf.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.minimum(
            tf.maximum(min_floor, tf.floor(queries)), max_floor)
        int_floor = tf.cast(floor, tf.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)


    flattened_grid = tf.reshape(grid,
                                       [batch_size * height * width, channels])
    batch_offsets = tf.reshape(
        tf.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name):
      with tf.name_scope('gather-' + name):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values,
                                 [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with tf.name_scope('interpolate'):
      interp_top = alphas[1] * (top_right - top_left) + top_left
      interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
      interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp

# def flow_warp(x, flow, interp_mode="bilinear", padding="sane"):
#     """Warp an image or feature map with optical flow
#
#     x (Tensor): size (N, H, W， C)
#     flow (Tensor): size (N, H, W, 2), normal value
#     interp_mode (str): 'nearest' or 'bilinear'
#     padding_mode (str): 'zeros' or 'border' or 'reflection'
#     Returns:
#         Tensor: warped image or feature map
#     """
#     shape = tf.shape(x)
#     batch_size = shape[0]
#     height = shape[1]
#     width = shape[2]
#     channels = shape[3]
#     grid_x, grid_y = tf.meshgrid(
#         tf.range(width), tf.range(height))
#     grid = tf.cast(
#         tf.stack([grid_y, grid_x], axis=2), flow.dtype)
#     batched_grid = tf.expand_dims(grid)
#     query_points_on_grid = batched_grid - flow
#     query_points_flattened = tf.reshape(query_points_on_grid,
#                                                [batch_size, height * width, 2])
#     interpolated = _interpolate_bilinear(image, query_points_flattened)
#     interpolated = tf.reshape(interpolated,
#                                      [batch_size, height, width, channels])
#     return interpolated

if __name__ == "__main__":
    import functools
    ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=64)
    model = Module(ResidualBlock_noBN_f, 3)
    input = tf.ones(shape=[64, 100, 100, 64])
    print(model(input))
