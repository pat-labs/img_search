FUNCTION Localize_And_Filter_Keypoints(candidate_keypoints, dog_pyramid):
    stable_keypoints = []
    FOR each candidate (x, y, octave, scale_index) in candidate_keypoints:

        // 1. Refine Location with Taylor Expansion
        //    Fit a 3D quadratic function to the local sample points to find the
        //    sub-pixel location of the extremum (x_hat, y_hat, sigma_hat).
        //    This gives a more precise location than the initial integer coordinates.
        offset, contrast_value = Fit_Quadratic_To_Extremum(dog_pyramid, x, y, octave, scale_index)

        // 2. Reject Low-Contrast Keypoints
        //    If the value at the refined extremum is close to zero, it's sensitive to noise.
        IF abs(contrast_value) < CONTRAST_THRESHOLD:
            CONTINUE // Reject this keypoint
        ENDIF

        // 3. Eliminate Edge Responses
        //    The DoG operator produces strong responses along edges, which are poorly
        //    localized. Use the ratio of principal curvatures (from the 2x2 Hessian matrix)
        //    to detect and reject these edge points. A high ratio indicates an edge.
        hessian_ratio = Calculate_Hessian_Ratio(dog_pyramid, x, y, octave, scale_index)
        IF hessian_ratio > EDGE_THRESHOLD_RATIO:
            CONTINUE // Reject this keypoint
        ENDIF

        // If the keypoint survived all tests, add it to the list with its precise location and scale.
        precise_location = (x, y) + offset.xy
        precise_scale = Calculate_Scale(octave, scale_index, offset.sigma)
        stable_keypoints.add( (precise_location, precise_scale) )
    ENDFOR

    RETURN stable_keypoints