import numpy as np


def corrupt_data(data, k):
  """Gives corrupted data for either one timestep or for a full year dataset
     depending on the input. k is the percent of corrupted data points.
     Expects a numpy array for the data input."""
  data_c = np.copy(data)
  m, n = 91, 180
  indices = np.random.choice(m*n, size=int(k*m*n), replace=False)

  # Convert the indices into row and column indices in the 2D meshgrid
  lat_indices, lon_indices = np.unravel_index(indices, (m, n))

  # Set the randomly selected points to 0
  if (np.size(data_c.shape) == 3):
    for i in range(2):
      data_c[i][lat_indices, lon_indices] = 0

  elif (np.size(data_c.shape) == 4):
    for i in range(data_c.shape[0]):
      for j in range(2):
        indices = np.random.choice(m*n, size=int(k*m*n), replace=False)
        # Convert the indices into row and column indices in the 2D meshgrid
        lat_indices, lon_indices = np.unravel_index(indices, (m, n))
        data_c[i][j][lat_indices, lon_indices] = 0

  return data_c