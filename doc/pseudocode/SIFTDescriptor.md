FUNCTION Generate_SIFT_Descriptor(image, keypoint):
    // Inputs:
    //   image: The original input image.
    //   keypoint: A data structure containing keypoint's (x, y) location,
    //             scale (σ), and dominant orientation (θ).

    // 1. Define Descriptor Parameters
    DESCRIPTOR_WIDTH = 4      // 4x4 grid of histograms
    HISTOGRAM_BINS = 8        // 8 orientation bins per histogram
    PATCH_SIZE = 16           // 16x16 pixel neighborhood around the keypoint

    // 2. Prepare the Image Patch
    //    - Get the Gaussian-blurred image corresponding to the keypoint's scale.
    //    - Create a 16x16 pixel window centered on the keypoint.
    //    - Rotate this patch so the keypoint's orientation (θ) points upwards.
    //      This makes the descriptor rotation-invariant.
    rotated_patch = Get_Oriented_Patch(image, keypoint.location, keypoint.scale, keypoint.orientation, PATCH_SIZE)

    // 3. Calculate Gradients
    //    For each pixel in the 16x16 rotated_patch, calculate its gradient:
    //    - magnitude(x, y) = sqrt( (dx)^2 + (dy)^2 )
    //    - orientation(x, y) = atan2(dy, dx)
    //    The orientation should be relative to the keypoint's main orientation.
    gradient_magnitudes, gradient_orientations = Calculate_Gradients(rotated_patch)

    // 4. Initialize the Descriptor Vector
    //    The final vector will be 4 * 4 * 8 = 128 elements.
    descriptor_vector = array of 128 zeros

    // 5. Create Histograms for Sub-regions
    //    Iterate through the 4x4 grid of sub-regions within the 16x16 patch.
    FOR sub_region_row from 0 to DESCRIPTOR_WIDTH - 1:
        FOR sub_region_col from 0 to DESCRIPTOR_WIDTH - 1:

            // Define the 4x4 pixel area for the current sub-region
            sub_patch = Get_Sub_Patch(rotated_patch, sub_region_row, sub_region_col)

            // Create an 8-bin histogram for this sub-region
            histogram = array of 8 zeros

            // Iterate through each of the 16 pixels in the 4x4 sub-patch
            FOR each pixel in sub_patch:
                // Get the pre-calculated gradient magnitude and orientation
                mag = gradient_magnitudes[pixel.x, pixel.y]
                ori = gradient_orientations[pixel.x, pixel.y]

                // Apply a Gaussian weighting to the magnitude to down-weight pixels
                // far from the keypoint's center. This increases robustness.
                weight = Gaussian_Weight(pixel.location, keypoint.location)
                weighted_mag = mag * weight

                // Add the weighted magnitude to the correct bin in the histogram.
                // Use trilinear interpolation: distribute the vote into adjacent
                // orientation, row, and column bins to avoid sudden changes.
                bin_index = floor(ori / (360 / HISTOGRAM_BINS))
                Add_To_Histogram_With_Interpolation(histogram, bin_index, weighted_mag)
            ENDFOR

            // Append this sub-region's 8-bin histogram to the main descriptor vector
            start_index = (sub_region_row * DESCRIPTOR_WIDTH + sub_region_col) * HISTOGRAM_BINS
            descriptor_vector[start_index : start_index + HISTOGRAM_BINS] = histogram
        ENDFOR
    ENDFOR

    // 6. Normalize the Descriptor Vector
    //    This makes the descriptor robust to changes in illumination.
    Normalize_To_Unit_Length(descriptor_vector)

    // 7. Clip Large Values and Re-normalize
    //    This handles non-linear illumination changes (e.g., saturation).
    FOR i from 0 to 127:
        descriptor_vector[i] = min(descriptor_vector[i], 0.2)
    ENDFOR
    Normalize_To_Unit_Length(descriptor_vector)

    RETURN descriptor_vector
