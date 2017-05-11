#include "gauss_pyr.h"
#include <opencv2/opencv.hpp>

gauss_pyr::gauss_pyr(cv::Mat& img, int _number_of_layers, float sigma)
{
    number_of_layers = _number_of_layers;
    cv::Mat dst = img.clone(); //deep copy to not modify original image
    for (int i = 0; i < number_of_layers; ++i)
    {
        cv::GaussianBlur(dst, dst, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
        layers_.push_back( dst.clone() ); // remember to deep copy!
        cv::resize(dst, dst, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
}

cv::Mat gauss_pyr::get(int layer) const
{
    return layers_[layer];
}

int gauss_pyr::getNumberOfLayers() const
{
    return number_of_layers;
}
