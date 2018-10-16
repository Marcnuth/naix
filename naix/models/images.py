from naix.algorithms.images import match_otsu_template


class TemplateContainsModel():
    """ Could be changed to a NN model if enough data
    """
    def predict(self, current_image, essential_tpls):
        for ele in essential_tpls:
            top_left, bottom_right, rmse, ssim = match_otsu_template(ele, current_image)
            if ssim > 0.95 and rmse < 0.2:
                return True
        return False
