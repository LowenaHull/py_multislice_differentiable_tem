"""Utility functions for working with the numpy library."""
import numpy as np
import torch
import copy


def ensure_array(input):
    """Force a (potentially scalar) input to be an array."""
    if hasattr(input, "__len__"):
        return input
    else:
        return np.asarray([input])


def q_space_array(pixels, gridsize):
    """
    Return the appropriately scaled 2D reciprocal space coordinates.

    Parameters
    -----------
    pixels : (2,) array_like
        Pixels in each dimension of a 2D array
    gridsize : (2,) array_like
        Dimensions of the array in real space units
    """
    qspace = [np.fft.fftfreq(pixels[i], d=gridsize[i] / pixels[i]) for i in [0, 1]]
    return [
        np.broadcast_to(q, pixels) for q in [qspace[0][:, None], qspace[1][None, :]]
    ]


def crop_window_to_flattened_indices(indices, shape):
    """Map  y and x indices describing a cropping window to a flattened 1d array."""
    return (indices[-1][None, :] + indices[-2][:, None] * shape[-1]).ravel()


def crop_to_bandwidth_limit(array, limit=2 / 3):
    """
    Crop an array to its bandwidth limit (ie remove superfluous array entries).

    assumes that input array is in Fourier space with zeroth Fourier component
    in upper-left corner
    """
    # Get array shape
    gridshape = array.shape[-2:]

    # New shape of final dimensions
    newshape = tuple([round(gridshape[i] * limit) for i in range(2)])

    # Indices of values to take
    ind = [
        (np.fft.fftfreq(newshape[i], 1 / newshape[i]).astype(np.int) + gridshape[i])
        % gridshape[i]
        for i in range(2)
    ]
    ind = crop_window_to_flattened_indices(ind, array.shape[-2:])

    # flatten final two dimensions of array
    flat_shape = array.shape[:-2] + (np.prod(array.shape[-2:]),)
    newshape = array.shape[:-2] + newshape

    return array.reshape(flat_shape)[..., ind].reshape(newshape)


def bandwidth_limit_array(arrayin, limit=2 / 3):
    """Band-width limit an array to fraction of its maximum given by limit."""
    # Case where band-width limiting has been turned off
    if limit is None:
        return arrayin

    array = copy.deepcopy(arrayin)
    if isinstance(array, np.ndarray):
        pixelsize = array.shape[:2]
        array[
            (
                np.square(np.fft.fftfreq(pixelsize[0]))[:, np.newaxis]
                + np.square(np.fft.fftfreq(pixelsize[1]))[np.newaxis, :]
            )
            * (2 / limit) ** 2
            > 1
        ] = 0
    else:
        pixelsize = array.size()[:2]
        array[
            (
                torch.from_numpy(np.fft.fftfreq(pixelsize[0]) ** 2).view(
                    pixelsize[0], 1
                )
                + torch.from_numpy(np.fft.fftfreq(pixelsize[1]) ** 2).view(
                    1, pixelsize[1]
                )
            )
            * (2 / limit) ** 2
            > 1
        ] = 0

    return array


def Fourier_interpolation_masks(npiyin, npixin, npiyout, npixout):
    """Calculate a mask of array entries to be included in Fourier interpolation."""
    # Construct input and output fft grids
    qyin, qxin, qyout, qxout = [
        (np.fft.fftfreq(x, 1 / x)).astype(np.int)
        for x in [npiyin, npixin, npiyout, npixout]
    ]

    # Get maximum and minimum common reciprocal space coordinates
    minqy, maxqy = [
        max(np.amin(qyin), np.amin(qyout)),
        min(np.amax(qyin), np.amax(qyout)),
    ]
    minqx, maxqx = [
        max(np.amin(qxin), np.amin(qxout)),
        min(np.amax(qxin), np.amax(qxout)),
    ]

    # Make 2d grids
    qqxout, qqyout = np.meshgrid(qxout, qyout)
    qqxin, qqyin = np.meshgrid(qxin, qyin)

    # Make a masks of common Fourier components for input and output arrays
    maskin = np.logical_and(
        np.logical_and(qqxin <= maxqx, qqxin >= minqx),
        np.logical_and(qqyin <= maxqy, qqyin >= minqy),
    )

    maskout = np.logical_and(
        np.logical_and(qqxout <= maxqx, qqxout >= minqx),
        np.logical_and(qqyout <= maxqy, qqyout >= minqy),
    )

    return maskin, maskout


def renormalize(array, newmax=1.0, newmin=0.0):
    """Rescales the array such that its maximum is newmax and its minimum is newmin."""
    max_, min_ = [array.max(), array.min()]
    return (array - min_) / (max_ - min_) * (newmax - newmin)


def colorize(z):
    """Map a complex number to the hsl scale and output in RGB format."""
    from colorsys import hls_to_rgb

    # Get phase an amplitude of complex array
    r = np.abs(z)
    arg = np.angle(z)

    # Calculate hue, lightness and saturation
    h = arg / (2 * np.pi)
    ell = renormalize(r, newmax=0.5)
    s = 0.8

    # Convert HLS format to RGB format
    c = np.vectorize(hls_to_rgb)(h, ell, s)  # --> tuple
    # Convert to numpy array
    c = np.array(c)  # -->
    # Array has shape (3,n,m), but we need (n,m,3) for output
    c = (c.swapaxes(0, 2) * 256).astype(np.uint8)
    return c


def fourier_interpolate_2d(ain, shapeout, norm="conserve_val"):
    """
    Perfom fourier interpolation on array ain so that its shape matches shapeout.

    Arguments
    ---------
    ain      -- array_like
        Input numpy array
    shapeout -- int (2,) , array_like
        Shape of output array

    Keyword arguments
    -----------------
    norm     -- str
        Normalization of output. If 'conserve_val' then array values are preserved
        if 'conserve_norm' L2 norm is conserved under interpolation and if
        'conserve_L1' L1 norm is conserved under interpolation
    """
    # Import required FFT functions
    from numpy.fft import fft2, ifft2

    # Make input complex
    aout = np.zeros(np.shape(ain)[:-2] + tuple(shapeout), dtype=np.complex)

    # Get input dimensions
    npiyin, npixin = np.shape(ain)[-2:]
    npiyout, npixout = shapeout

    # Get Fourier interpolation masks
    maskin, maskout = Fourier_interpolation_masks(npiyin, npixin, npiyout, npixout)

    # Now transfer over Fourier coefficients from input to output array
    aout[..., maskout] = fft2(np.asarray(ain, dtype=np.complex))[..., maskin]

    # Fourier transform result with appropriate normalization
    if norm == "conserve_val":
        aout = ifft2(aout) * (np.prod(shapeout) / np.prod(np.shape(ain)[-2:]))
    elif norm == "conserve_norm":
        aout = ifft2(aout) * np.sqrt(np.prod(shapeout) / np.prod(np.shape(ain)[-2:]))
    else:
        aout = ifft2(aout)

    # Return correct array data type
    if (
        str(ain.dtype) in ["float64", "float32", "float", "f"]
        or str(ain.dtype)[1] == "f"
    ):
        return np.real(aout)
    else:
        return aout


def oned_shift(N, shift, pixel_units=True):
    """
    Construct a one-dimensional shift array.

    Parameters
    ----------
    N     -- Number of pixels in the shift array
    shift -- Amount of shift to be achieved (default units of pixels)

    Keyword arguments
    ----------
    pixel_units -- Pass True if shift is to be units of pixels, False for
                   fraction of the array
    """
    # Create the Fourier space pixel coordinates of the shift array
    shiftarray = (np.arange(N) + N // 2) % N - N // 2

    # Conversion necessary if the shift is in units of pixels, and not fractions
    # of the array
    if pixel_units:
        shiftarray = shiftarray / N

    # The shift array is given mathematically as e^(-2pi i k Delta x) and this
    # is what is returned.
    return np.exp(-2 * np.pi * 1j * shiftarray * shift)


def fourier_shift(arrayin, shift, qspacein=False, qspaceout=False, pixel_units=True):
    """
    Shifts a 2d array using the Fourier shift theorem.

    Parameters
    ----------
    arrayin -- Array to be Fourier shifted
    shift   -- Shift in units of pixels (pass pixel_units = False for shift
               to be in units of fraction of the array size)

    Keyword arguments
    ----------
    qspacein    -- Pass True if arrayin is in Fourier space
    qspaceout   -- Pass True for Fourier space output, False (default) for
                   real space output
    pixel_units -- Pass True if shift is to be units of pixels, False for
                   fraction of the array
    """
    # Construct shift array
    shifty, shiftx = [
        oned_shift(arrayin.shape[-2 + i], shift[i], pixel_units) for i in range(2)
    ]

    # Now Fourier transform array and apply shift
    real = not np.iscomplexobj(arrayin)

    if real:
        array = np.asarray(arrayin, dtype=np.complex)
    else:
        array = arrayin

    if not qspacein:
        array = np.fft.fft2(array)

    array = shiftx[np.newaxis, :] * shifty[:, np.newaxis] * array

    if not qspaceout:
        array = np.fft.ifft2(array)

    if real:
        return np.real(array)
    else:
        return array


def crop(arrayin, shapeout):
    """
    Crop the last two dimensions of arrayin to grid size shapeout.

    For entries of shapeout which are larger than the shape of the input array,
    perform zero-padding.
    """
    # Number of dimensions in input array
    ndim = arrayin.ndim

    # Number of dimensions not covered by shapeout (ie not to be cropped)
    nUntouched = ndim - 2

    # Shape of output array
    shapeout_ = arrayin.shape[:nUntouched] + tuple(shapeout)

    arrayout = np.zeros(shapeout_, dtype=arrayin.dtype)

    y, x = arrayin.shape[-2:]
    y_, x_ = shapeout[-2:]

    def indices(y, y_):
        if y > y_:
            # Crop in y dimension
            y1, y2 = [(y - y_) // 2, (y + y_) // 2]
            y1_, y2_ = [0, y_]
        else:
            # Zero pad in y dimension
            y1, y2 = [0, y]
            y1_, y2_ = [(y_ - y) // 2, (y + y_) // 2]
        return y1, y2, y1_, y2_

    y1, y2, y1_, y2_ = indices(y, y_)
    x1, x2, x1_, x2_ = indices(x, x_)

    arrayout[..., y1_:y2_, x1_:x2_] = arrayin[..., y1:y2, x1:x2]
    return arrayout
