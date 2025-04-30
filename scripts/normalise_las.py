# Utility function written largely by ChatGPT to normalize a .las file instead of using 3DFin to normalize the .las file as that normally results in Potential Normalization Error
# to use:
# - set the various paths
# - conda activate 3dfin
# - run the code
import laspy
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
import rasterio
from rasterio.transform import from_origin

# Paths and parameters
# input_path  = "/Users/andrewcottam/Documents/QGIS/Data/restor/lidar/Tahi/Tahi_Area2_Full_sample2.las"
# output_path = "/Users/andrewcottam/Documents/QGIS/Data/restor/lidar/Tahi/Tahi_Area2_Full_sample2_z0.las"
# dtm_path    = "/Users/andrewcottam/Documents/QGIS/Data/restor/lidar/Tahi/Tahi_Area2_Full_sample2_dtm_from_python.tif"
input_path  = "/home/andrew/Downloads/Data/tahi/lidar/Tahi_Area2_LiDAR_Full.las"
output_path = "/home/andrew/Downloads/Data/tahi/lidar/Tahi_Area2_LiDAR_Full_z0.las"
dtm_path    = "/home/andrew/Downloads/Data/tahi/lidar/Tahi_Area2_LiDAR_Full_dtm.tif"
resolution  = 1.0  # grid cell size in same units as LAS coordinates

# 1. Read the LAS file
las = laspy.read(input_path)
x, y, z = las.x, las.y, las.z

# 2. Extract ground points (assumes classification is present and ground = class 2)
if "classification" not in las.point_format.dimension_names:
    raise ValueError("Input LAS has no 'classification' dimension. Please classify ground points first.")
ground_mask = las.classification == 2
xg, yg, zg = x[ground_mask], y[ground_mask], z[ground_mask]

# 3. Build regular grid over the extents
min_x, max_x = x.min(), x.max()
min_y, max_y = y.min(), y.max()
xi = np.arange(min_x, max_x + resolution, resolution)
yi = np.arange(min_y, max_y + resolution, resolution)
grid_x, grid_y = np.meshgrid(xi, yi)

# 4. Interpolate ground elevations onto the grid (linear, then nearest for holes)
grid_z = griddata((xg, yg), zg, (grid_x, grid_y), method="linear")
nan_mask = np.isnan(grid_z)
if np.any(nan_mask):
    grid_z[nan_mask] = griddata(
        (xg, yg), zg,
        (grid_x[nan_mask], grid_y[nan_mask]),
        method="nearest"
    )

# 5. Save the DTM as a GeoTIFF
transform = from_origin(min_x, max_y, resolution, resolution)
with rasterio.open(
    dtm_path, "w",
    driver="GTiff",
    height=grid_z.shape[0],
    width=grid_z.shape[1],
    count=1,
    dtype=grid_z.dtype,
    crs=las.header.parse_crs().to_wkt() if las.header.parse_crs() else None,
    transform=transform,
) as dst:
    dst.write(grid_z, 1)

# 6. Create interpolator and sample DTM under each point
interp = RegularGridInterpolator((yi, xi), grid_z, bounds_error=False, fill_value=None)
dtm_vals = interp(np.vstack((y, x)).T)

# 7. Compute normalized height (z0)
z0 = z - dtm_vals

# ── HERE’S THE FIX ──
# 8. Add your new “z0” field *via the LasData API*, not just the header:
extra_dim = laspy.ExtraBytesParams(name="z0", type=np.float32)
las.add_extra_dim(extra_dim)

# 9. Assign and write
las.z0 = z0.astype(np.float32)
las.write(output_path)

print(f"DTM saved to: {dtm_path}")
print(f"Normalized LAS with 'z0' field saved to: {output_path}")