FUNCTION Assign_Orientations(stable_keypoints, gaussian_pyramid):
    final_keypoints = []
    FOR each keypoint (location, scale) in stable_keypoints:
        // 1. Select the corresponding Gaussian-blurred image for the keypoint's scale.
        image_at_scale = Get_Image_From_Pyramid(gaussian_pyramid, keypoint.scale)

        // 2. Create an Orientation Histogram
        //    - Define a circular region around the keypoint.
        //    - For each pixel in this region, calculate its gradient magnitude and orientation.
        //    - Weight the magnitudes by a Gaussian window centered on the keypoint.
        //    - Add the weighted magnitudes to a 36-bin orientation histogram (10 degrees per bin).
        orientation_histogram = Create_Orientation_Histogram(image_at_scale, keypoint.location, keypoint.scale)

        // 3. Find the Dominant Orientation(s)
        //    The highest peak in the histogram is the keypoint's dominant orientation.
        dominant_orientation = Find_Peak(orientation_histogram)

        // 4. Create Keypoints for Each Major Orientation
        //    Any other peak within 80% of the highest peak's value also creates a new
        //    keypoint. This allows a single location to represent multiple features
        //    (e.g., a corner with two different textures).
        other_orientations = Find_Peaks_Above_Threshold(orientation_histogram, 0.8 * dominant_orientation.value)

        // Add the primary keypoint
        final_keypoints.add( Keypoint(location, scale, dominant_orientation.angle) )

        // Add any secondary keypoints
        FOR each secondary_orientation in other_orientations:
            final_keypoints.add( Keypoint(location, scale, secondary_orientation.angle) )
        ENDFOR
    ENDFOR

    RETURN final_keypoints