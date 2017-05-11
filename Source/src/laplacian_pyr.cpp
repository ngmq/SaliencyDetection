#include "laplacian_pyr.h"
#include <opencv2/opencv.hpp>

laplacian_pyr::laplacian_pyr(const gauss_pyr& p, float sigma)
{
    number_of_layers = p.getNumberOfLayers();
    for (int i = 0; i < number_of_layers; ++i)
    {
        cv::Mat src = p.get(i);
        cv::Mat dst = src.clone();
        cv::GaussianBlur(src, dst, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
        dst = src - dst;
        layers_.push_back( dst.clone() ); // remember to deep copy!
    }
}

cv::Mat laplacian_pyr::get(int layer) const
{
    return layers_[layer];
}

int laplacian_pyr::getNumberOfLayers() const
{
    return number_of_layers;
}
