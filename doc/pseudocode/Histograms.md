FUNCTION Create_Orientation_Histogram(image_at_scale, keypoint_location, keypoint_scale):
    // --- INPUTS ---
    // image_at_scale: The Gaussian-blurred image corresponding to the keypoint's scale.
    // keypoint_location: The precise (x, y) coordinates of the keypoint.
    // keypoint_scale: The scale (σ) of the keypoint, which determines the region size.

    // --- 1. INITIALIZATION ---
    NUM_BINS = 36
    HISTOGRAM = array of NUM_BINS zeros  // One bin for every 10 degrees (360 / 10)

    // The size of the region around the keypoint is proportional to its scale.
    // A typical radius is 3 times a factor (1.5) of the keypoint's scale.
    region_radius = floor(3 * 1.5 * keypoint_scale)
    
    // A Gaussian weighting function gives more importance to pixels closer to the center.
    // The standard deviation of this Gaussian is 1.5 times the keypoint's scale.
    gaussian_sigma = 1.5 * keypoint_scale
    two_gaussian_sigma_sq = 2 * gaussian_sigma * gaussian_sigma

    // --- 2. ITERATE THROUGH PIXELS IN THE REGION ---
    // Loop through a square window and later check if pixels are within the circular radius.
    FOR y_offset from -region_radius to +region_radius:
        FOR x_offset from -region_radius to +region_radius:
            
            // Get the actual pixel coordinates in the image
            pixel_x = keypoint_location.x + x_offset
            pixel_y = keypoint_location.y + y_offset

            // Ensure the pixel is within the image boundaries
            IF pixel is outside image_at_scale.bounds:
                CONTINUE
            ENDIF

            // --- 3. CALCULATE GRADIENT FOR THE PIXEL ---
            // Use finite differences to approximate the gradient
            dx = image_at_scale[pixel_x + 1, pixel_y] - image_at_scale[pixel_x - 1, pixel_y]
            dy = image_at_scale[pixel_x, pixel_y + 1] - image_at_scale[pixel_x, pixel_y - 1]

            gradient_magnitude = sqrt(dx*dx + dy*dy)
            gradient_orientation = atan2(dy, dx) // Result in radians, convert to degrees

            // --- 4. WEIGHT THE MAGNITUDE ---
            // Calculate the Gaussian weight based on distance from the keypoint center.
            // This down-weights the influence of pixels far from the keypoint.
            distance_sq = x_offset*x_offset + y_offset*y_offset
            gaussian_weight = exp(-distance_sq / two_gaussian_sigma_sq)
            
            weighted_magnitude = gradient_magnitude * gaussian_weight

            // --- 5. ADD TO HISTOGRAM ---
            // Determine which bin the orientation falls into.
            bin_index = floor(gradient_orientation_in_degrees / 10.0) % NUM_BINS
            
            // Add the weighted magnitude to the corresponding bin.
            HISTOGRAM[bin_index] += weighted_magnitude
            
        ENDFOR
    ENDFOR

    // --- 6. SMOOTH THE HISTOGRAM ---
    // To make it more robust, smooth the histogram by averaging each bin with its neighbors.
    // This can be done by convolving with a small Gaussian kernel or a simple moving average.
    smoothed_histogram = Smooth(HISTOGRAM)

    RETURN smoothed_histograms