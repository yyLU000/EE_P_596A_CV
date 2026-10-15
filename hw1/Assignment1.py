import cv2
import numpy as np
import os


class ComputerVisionAssignment:
    def __init__(self, image_path, binary_image_path):
        self.image = cv2.imread(image_path)
        self.binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    def check_package_versions(self):
        # Ungraded
        import numpy as np
        import matplotlib
        import cv2

        # print(np.__version__)
        # print(matplotlib.__version__)
        # print(cv2.__version__)

    def load_and_analyze_image(self):
        """
        Fill your code here

        """

        image_path = "original_image.png"
        self.image = cv2.imread(image_path)

        # Print the data type of the image, and the data type of the pixels in the image.
        Image_data_type = type(self.image)
        Pixel_data_type = self.image.dtype

        # Print the image dimensions.
        Image_shape = self.image.shape

        # Check if the datatype of the image and a pixel in the image are in uint8 format.
        # print("Image stores uint8 values:", self.image.dtype == np.uint8)
        # print("Pixel channel is uint8:", isinstance(self.image[0, 0, 0], np.uint8))

        # Return the data types and image dimensions.
        # print(f"Image data type: {Image_data_type}")
        # print(f"Pixel data type: {Pixel_data_type}")
        # print(f"Image dimensions: {Image_shape}")

        return Image_data_type, Pixel_data_type, Image_shape

    def create_red_image(self):
        """
        Fill your code here

        """

        red_image = self.image.copy()
        red_image[:, :, 0] = 0  # Blue
        red_image[:, :, 1] = 0  # Green

        return red_image

    def create_photographic_negative(self):
        """
        Fill your code here

        """

        negative_image = 255 - self.image

        return negative_image

    def swap_color_channels(self):
        """
        Fill your code here

        """

        swapped_image = self.image[:, :, [2, 1, 0]] # RED <-> BLUE

        return swapped_image

    def foliage_detection(self):
        """
        Fill your code here

        """

        blue = self.image[:, :, 0]
        green = self.image[:, :, 1]
        red = self.image[:, :, 2]

        mask = (green >= 50) & (blue < 50) & (red < 50)
        foliage_image = mask.astype(np.uint8) * 255

        return foliage_image

    def shift_image(self):
        """
        Fill your code here

        """

        shifted_image = np.zeros_like(self.image)
        shifted_image[100:, 200:] = self.image[:-100, :-200]

        return shifted_image

    def rotate_image(self):
        """
        Fill your code here

        """

        rotated_image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        return rotated_image

    def similarity_transform(self, scale, theta, shift):
        """
        Fill your code here

        """

        h, w = self.image.shape[:2]
        transformed_image = np.zeros_like(self.image)

        angle = np.deg2rad(theta)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        dx, dy = shift

        for Y in range(h):
            for X in range(w):
                # Undo translation
                x = X - dx
                y = Y - dy

                # Undo rotation, then scaling
                x_src = (cos_theta * x + sin_theta * y) / scale
                y_src = (-sin_theta * x + cos_theta * y) / scale

                # Nearest-neighbor interpolation.
                x_src = int(round(x_src))
                y_src = int(round(y_src))

                if 0 <= x_src < w and 0 <= y_src < h :
                    transformed_image[Y, X] = self.image[y_src, x_src]

        return transformed_image

    def convert_to_grayscale(self):
        """
        Fill your code here

        """

        blue = self.image[:, :, 0].astype(np.float64)
        green = self.image[:, :, 1].astype(np.float64)
        red = self.image[:, :, 2].astype(np.float64)
        gray_image = np.round( (3 * red + 6 * green + 1 * blue) / 10 ).astype(np.uint8)

        # cv2.imwrite("grey_image.png", gray_image)
        return gray_image

    def compute_moments(self):
        """
        Fill your code here

        """
        binary = self.binary_image.astype(np.float64)
        y, x = np.indices(binary.shape)

        m00 = np.sum(binary)
        if m00 == 0:
            raise ValueError("The image has no foreground pixels.")
        
        m10 = np.sum(x * binary)
        m01 = np.sum(y * binary)

        x_bar = m10 / m00
        y_bar = m01 / m00

        dx = x - x_bar
        dy = y - y_bar

        mu20 = np.sum(dx**2 * binary)
        mu02 = np.sum(dy**2 * binary)
        mu11 = np.sum(dx * dy * binary)

        # Print the results
        # print("First-Order Moments:")
        # print(f"Standard (Raw) Moments: M00 = {m00}, M10 = {m10}, M01 = {m01}")
        # print("Centralized Moments:")
        # print(f"x_bar = {x_bar}, y_bar = {y_bar}")
        # print("Second-Order Centralized Moments:")
        # print(f"mu20 = {mu20}, mu02 = {mu02}, mu11 = {mu11}")

        return m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11

    def compute_orientation_and_eccentricity(self):
        """
        Fill your code here

        """

        m00, m10, m01, x_bar, y_bar, mu20, mu02, mu11 = self.compute_moments()
        orientation = np.degrees(0.5 * np.arctan(2 * mu11 / (mu20 - mu02)))
        a = mu20 / m00
        b = mu11 / m00
        c = mu02 / m00
        diff = np.sqrt((a - c)**2 + 4 * b**2)
        lambda1 = (a + c + diff) / 2
        lambda2 = (a + c - diff) / 2
        eccentricity = np.sqrt((lambda1 - lambda2) / lambda1)

        full_major_axis = 4 * np.sqrt(lambda1)
        full_minor_axis = 4 * np.sqrt(lambda2)
        half_axes = (int(full_major_axis // 2), int(full_minor_axis // 2))
        center = (int(x_bar), int(y_bar))
        glasses_with_ellipse = self.image.copy()

        cv2.ellipse(
            glasses_with_ellipse, 
            center,
            half_axes,
            float(orientation),
            0,
            360,
            (0, 0, 255),
            1
        )

        return orientation, eccentricity, glasses_with_ellipse


if __name__ == "__main__":

    assignment = ComputerVisionAssignment("original_image.png", "binary_image.png")

    # Task 0: Check package versions
    assignment.check_package_versions()

    # Task 1: Load and analyze the image
    assignment.load_and_analyze_image()

    # Task 2: Create a red image
    red_image = assignment.create_red_image()

    # Task 3: Create a photographic negative
    negative_image = assignment.create_photographic_negative()

    # Task 4: Swap color channels
    swapped_image = assignment.swap_color_channels()

    # Task 5: Foliage detection
    foliage_image = assignment.foliage_detection()

    # Task 6: Shift the image
    shifted_image = assignment.shift_image()

    # Task 7: Rotate the image
    rotated_image = assignment.rotate_image()

    # Task 8: Similarity transform
    transformed_image = assignment.similarity_transform(
        scale=2.0, theta=45.0, shift=[100, 100]
    )
    # cv2.imwrite("transformed_image.png", transformed_image)

    # Task 9: Grayscale conversion
    gray_image = assignment.convert_to_grayscale()

    glasses_assignment = ComputerVisionAssignment(
        "glasses_outline.png", "glasses_outline.png"
    )

    # Task 10: Moments of a binary image
    glasses_assignment.compute_moments()

    # Task 11: Orientation and eccentricity of a binary image
    orientation, eccentricity, glasses_with_ellipse = (
        glasses_assignment.compute_orientation_and_eccentricity()
    )
    # cv2.imwrite("glasses_with_ellipse.png", glasses_with_ellipse)