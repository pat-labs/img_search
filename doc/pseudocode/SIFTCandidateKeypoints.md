FUNCTION Find_Candidate_Keypoints(image, num_octaves, scales_per_octave, initial_sigma):
    // 1. Build Gaussian Pyramid
    //    Create a series of increasingly blurred images (octaves).
    //    Within each octave, create images blurred by increasing amounts (scales).
    gaussian_pyramid = Build_Gaussian_Pyramid(image, num_octaves, scales_per_octave, initial_sigma)

    // 2. Build Difference-of-Gaussians (DoG) Pyramid
    //    Subtract adjacent blurred images in the Gaussian pyramid. This approximates
    //    the Laplacian of Gaussian, which is great for finding blobs.
    dog_pyramid = Build_DoG_Pyramid(gaussian_pyramid)

    // 3. Find Local Extrema in Scale-Space
    //    Iterate through each pixel of the DoG images and compare it to its
    //    26 neighbors: 8 neighbors in the same image, and 9 neighbors each
    //    in the images above and below it in scale.
    candidate_keypoints = []
    FOR each octave in dog_pyramid:
        FOR each scale_index from 1 to scales_per_octave:
            current_dog_image = dog_pyramid[octave][scale_index]
            prev_dog_image    = dog_pyramid[octave][scale_index - 1]
            next_dog_image    = dog_pyramid[octave][scale_index + 1]

            FOR each pixel (x, y) in current_dog_image (excluding borders):
                pixel_value = current_dog_image[x, y]

                // Check if the pixel is a local maximum or minimum in the 3x3x3 neighborhood
                is_maxima = Is_Greater_Than_All_26_Neighbors(pixel_value, ...)
                is_minima = Is_Less_Than_All_26_Neighbors(pixel_value, ...)

                IF is_maxima OR is_minima:
                    // This is a candidate keypoint, stable across scales.
                    candidate_keypoints.add( (x, y, octave, scale_index) )
                ENDIF
            ENDFOR
        ENDFOR
    ENDFOR

    RETURN candidate_keypoints