#include "oriented_pyr.h"
#include <opencv2/opencv.hpp>

oriented_pyr::oriented_pyr(const laplacian_pyr& p, int num_orientations)
{
    number_of_layers = p.getNumberOfLayers();
    number_of_orientations = num_orientations;
    orientation_maps.resize(number_of_orientations);
    double step = CV_PI / number_of_orientations;

    std::vector<cv::Mat> gaborFilters;
    for(int i = 0; i < num_orientations; ++i)
    {
        double theta = -CV_PI / 2 - i * step;
        cv::Mat kernel = cv::getGaborKernel( cv::Size(25, 25), 10.0, theta, 10.0, 1.0, 0, CV_32F);
        gaborFilters.push_back(kernel.clone());
    }
    for(int i = 0; i < number_of_orientations; ++i)
    {
        for(int j = 0; j < number_of_layers; ++j)
        {
            cv::Mat src = p.get(j);
            cv::Mat dst;
            cv::filter2D(src, dst, CV_32F, gaborFilters[i]);
            orientation_maps[i].push_back(dst.clone());
        }  
    }
}

std::vector <cv::Mat > oriented_pyr::getByOrientation(int orientation) const
{
    return orientation_maps[orientation];
}

std::vector <cv::Mat > oriented_pyr::getByLayer( int layer) const
{
    std::vector <cv::Mat > res;
    for(int i = 0; i < number_of_orientations; ++i) res.push_back(orientation_maps[i][layer]);
    return res;
}

cv::Mat oriented_pyr::get(int orientation, int layer) const
{
    return orientation_maps[orientation][layer];
}
