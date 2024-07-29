import numpy as np

def ParabolaMask(GridPositionMatrix_X_out, GridPositionMatrix_Y_out, coef):
    """
    Generates a parabolic phase mask.

    Args:
        GridPositionMatrix_X_out (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_out (np.ndarray): 2D array with the y coordinates of the grid (in m).
        coef (float): Coefficient for the parabolic equation (in m^-1).
        position (tuple): Position of the vertex of the parabola (in m).

    Returns:
        np.ndarray: 2D array with the mask.
    """

    # Calculate the parabola surface z value
    Z_out = coef * (GridPositionMatrix_X_out ** 2 + GridPositionMatrix_Y_out ** 2)

    return Z_out



def supergaussian2D(x_array_in, n, sigma):
    """
    Generate a super-gaussian function of order n in 2D.

    Parameters:
        x (ndarray): x-coordinates of the grid.
        y (ndarray): y-coordinates of the grid.
        n (int): Order of the super-gaussian function.
        sigma (float): Standard deviation of the function.

    Returns:
        ndarray: The values of the super-gaussian function evaluated at the input coordinates.
    """

    x, y = np.meshgrid(x_array_in, x_array_in)

    x2 = x ** 2
    y2 = y ** 2
    r2 = x2 + y2
    return np.exp(-((r2 / sigma ** 2) ** (n / 2)))



def PhaseReversalMask(GridPositionMatrix_X_in,GridPositionMatrix_Y_in,input_waist,sigma_x,sigma_y):

    """
    Generates a phase reversal mask.

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        input_waist (float): Input waist of the beam (in m).
        sigma_x (float): Sigma of the beam in the x direction (in m).
        sigma_y (float): Sigma of the beam in the y direction (in m).

    Returns:
        np.ndarray: 2D array with the mask.

    """

    sinc_step_x = input_waist*sigma_x
    sinc_step_y = input_waist*sigma_y
    depth_param = np.pi

    sinc_mask_x = sinc_resized(GridPositionMatrix_X_in,sinc_step_x)
    sinc_mask_y = sinc_resized(GridPositionMatrix_Y_in,sinc_step_y)
    sinc_mask = sinc_mask_x*sinc_mask_y

    M = np.zeros(GridPositionMatrix_X_in.shape)
    M[sinc_mask < 0] = 1
    M *= depth_param

    return M

def RectangularMask(GridPositionMatrix_X_in, GridPositionMatrix_Y_in, angle, width, height, position=(0, 0)):
    """
    Generates a rectangular amplitude mask with an optional position.

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        angle (float): Rotation angle of the rectangle (in rad).
        width (float): Width of the rectangle (in m).
        height (float): Height of the rectangle (in m).
        position (tuple): Center position of the rectangle (in m).

    Returns:
        np.ndarray: 2D array with the mask.
    """

    # Unpack the position tuple
    x0, y0 = position

    # Shift the grid coordinates by the position
    shifted_X = GridPositionMatrix_X_in - x0
    shifted_Y = GridPositionMatrix_Y_in - y0

    # Rotate the grid
    rotated_X = shifted_X * np.cos(angle) - shifted_Y * np.sin(angle)
    rotated_Y = shifted_Y * np.cos(angle) + shifted_X * np.sin(angle)

    # Initialize the mask
    mask = np.zeros(GridPositionMatrix_X_in.shape)

    # Apply the mask
    mask[(np.abs(rotated_X) < width / 2) & (np.abs(rotated_Y) < height / 2)] = 1

    return mask

def SinusMask(GridPositionMatrix_X_in, GridPositionMatrix_Y_in, period, angle, phase_offset=0):

    """
    Generates a sinusoidal amplitude array (sinus).

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        period (float): Period of the sinusoidal amplitude array (in m).
        angle (float): Rotation angle of the sinusoidal amplitude array (in rad).
        phase_offset (float): Phase offset of the sinusoidal amplitude array (in rad).

    Returns:
        np.ndarray: 2D array with the mask.

    """

    GridPositionMatrix_X_in_rot = GridPositionMatrix_X_in * np.cos(angle) - GridPositionMatrix_Y_in * np.sin(angle)
    mask = np.sin(2 * np.pi * GridPositionMatrix_X_in_rot / period + phase_offset)
    return mask


def CosinusMask(GridPositionMatrix_X_in, GridPositionMatrix_Y_in, period, angle):

    """
    Generates a sinusoidal amplitude array (cosinus).

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        period (float): Period of the sinusoidal amplitude array (in m).
        angle (float): Rotation angle of the sinusoidal amplitude array (in rad).

    Returns:
        np.ndarray: 2D array with the mask.

    """
    GridPositionMatrix_X_in_rot = GridPositionMatrix_X_in * np.cos(angle) - GridPositionMatrix_Y_in * np.sin(angle)

    mask = np.cos(2 * np.pi * GridPositionMatrix_X_in_rot / period)
    return mask

def PiPhaseJumpMask(GridPositionMatrix_X_in, GridPositionMatrix_Y_in, orientation, position):

    """
    Generates a pi phase jump mask.

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        orientation (str): Orientation of the pi phase jump mask (either "Vertical" or "Horizontal").
        position (float): Position of the pi phase jump mask (in m).

    Returns:
        np.ndarray: 2D array with the mask.

    """

    mask = np.zeros(GridPositionMatrix_X_in.shape)

    if orientation == "Vertical":
        mask[GridPositionMatrix_Y_in > position] = np.pi
    elif orientation == "Horizontal":
        mask[GridPositionMatrix_X_in > position] = np.pi

    return mask

def Simple2DBlazedGratingMask(GridPositionMatrix_X_in, GridPositionMatrix_Y_in, period, angles):

    """
    Generates a blazed grating mask with three sections, each with a different angle.
    The mask is generated by rotating the grid and then selecting the regions with the desired angle.

    Args:
        GridPositionMatrix_X_in (np.ndarray): 2D array with the x coordinates of the grid (in m).
        GridPositionMatrix_Y_in (np.ndarray): 2D array with the y coordinates of the grid (in m).
        period (float): Period of the grating (in m)
        angles (list): List with the three angles of the grating (in rad).

    Returns:
        np.ndarray: 2D array with the mask.

    """


    assert len(angles) == 3, "Three angles required for three sections of the mask."

    # Calculate grid sizes
    grid_size_x = GridPositionMatrix_X_in[-1, -1] - GridPositionMatrix_X_in[0, 0]
    grid_size_y = GridPositionMatrix_Y_in[-1, -1] - GridPositionMatrix_Y_in[0, 0]

    # Initialize the mask
    mask = np.zeros(GridPositionMatrix_X_in.shape)

    # Create a distance matrix representing the distance from the center
    GridDistanceMatrix = np.sqrt(GridPositionMatrix_X_in ** 2 + GridPositionMatrix_Y_in ** 2)
    max_radius = np.sqrt(grid_size_x ** 2 + grid_size_y ** 2) / 2  # maximum radius covering the grid

    # Create angle array to determine the region
    angle_array = np.arctan2(GridPositionMatrix_Y_in, GridPositionMatrix_X_in)
    angle_array = (angle_array + np.pi) % (2 * np.pi)  # convert range from [-pi, pi] to [0, 2*pi]

    # Generate the three grating masks
    grating_masks = []
    for angle in angles:
        # Create rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        # Rotate grid positions
        rotated_positions = np.einsum('ji, mni -> jmn', rotation_matrix,
                                      np.dstack([GridPositionMatrix_X_in, GridPositionMatrix_Y_in]))
        rotated_X = rotated_positions[0, :, :]
        grating_masks.append(rotated_X % period)

    # Use circular masks to select regions from the grating masks
    for i in range(3):
        mask[((angle_array >= i * 2 * np.pi / 3) & (angle_array < (i + 1) * 2 * np.pi / 3) & (
                    GridDistanceMatrix <= max_radius))] = grating_masks[i][(
                    (angle_array >= i * 2 * np.pi / 3) & (angle_array < (i + 1) * 2 * np.pi / 3) & (
                        GridDistanceMatrix <= max_radius))]

    return mask



def SlitMask(x_array,x0,width):
    """
    Generates a 1D slit mask.

    Args:
        x_array (np.ndarray): 1D array with the x coordinates of the grid (in m).
        x0 (float): Center of the slit (in m).

    Returns:
        np.ndarray: 1D array with the mask.

    """

    mask = np.ones(x_array.shape[0])
    mask[x_array<x0-width/2] = 0
    mask[x_array>=x0+width/2] = 0
    return mask


def Simple1DGratingMask(x_array,period):

    """
    Generates a 1D grating mask.

    Args:
        x_array (np.ndarray): 1D array with the x coordinates of the grid (in m).
        period (float): Period of the grating (in m).

    Returns:
        np.ndarray: 1D array with the mask.

    """

    sampling = x_array.shape[0]
    grid_size = x_array[-1] - x_array[0]
    mask = np.zeros(sampling)
    number_of_periods = int(grid_size / period)

    print("period = ",period)
    print("number_of_periods = ",number_of_periods)
    print("grid_size = ",grid_size)
    print("sampling = ",sampling)

    for per in range(number_of_periods):
        # create a mask numpy array to select x values inside the period range
        min_x_per = per * period - grid_size / 2
        max_x_per = (per + 1) * period - grid_size / 2
        masking_array = (x_array >= min_x_per) & (x_array < max_x_per)

        if per % 2 == 0:
            mask[masking_array] += 1
    return mask

def VortexMask(x_array,charge):

    """
    Generates a vortex mask.

    Args:
        x_array (np.ndarray): 1D array with the x coordinates of the grid (in m).
        charge (int): Charge of the vortex.

    """

    x, y = np.meshgrid(x_array, x_array)

    mask = -charge * np.arctan2(y, x)

    return mask

def Simple1DBlazedGratingMask(x_array, period):

    """
    Generates a 1D blazed grating mask.

    Args:
        x_array (np.ndarray): 1D array with the x coordinates of the grid (in m).
        period (float): Period of the grating (in m).

    Returns:
        np.ndarray: 1D array with the mask.


    """

    dimension = x_array.shape[0]
    # Generate one period of the ramp function
    ramp_ = np.linspace(-np.pi, np.pi, period+1)
    ramp = ramp_[:-1]

    # calculate how many full periods and extra points we will need


    full_periods = int(dimension // period)
    extra_points = int(dimension % period)

    # create the mask for the full periods
    full_mask = np.tile(ramp, full_periods)
    # add extra points if needed
    if extra_points > 0:
        extra_mask = ramp[:extra_points]
        mask = np.concatenate([full_mask, extra_mask])
    else:
        mask = full_mask

    return mask


def Simple2DWedgeMask(x_array,wavelength,x_position,focal_length):

    """
    Generates a 2D wedge mask.

    Args:
        x_array (np.ndarray): 1D array with the x coordinates of the grid (in m).
        wavelength (float): Wavelength of the light (in m).
        x_position (float): Position of the wedge (in m).
        focal_length (float): Focal length of the lens (in m).

    Returns:
        np.ndarray: 2D array with the mask.

    """

    angle = np.arctan(x_position/focal_length)

    max_phase = 2*np.pi*angle / np.arctan(wavelength/(x_array.max() - x_array.min()))
    wedge_1D = np.linspace(0, max_phase, x_array.shape[0])
    wedge_2D = np.tile(wedge_1D, ( x_array.shape[0],1))

    return wedge_2D

def sinc_resized(x,step):
    """
    Resizes the sinc function to a given step size.

    Args:
        x (np.ndarray): 1D array with the x coordinates of the grid (in m).
        step (float): Step size (in m).

    Returns:
        np.ndarray: 1D array with the resized sinc function.
    """

    return np.sinc(x/step)