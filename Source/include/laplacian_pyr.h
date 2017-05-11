#include <vector>
#include <opencv2/opencv.hpp>
#include "gauss_pyr.h"

class laplacian_pyr
{
public:
    laplacian_pyr(const gauss_pyr& p, float sigma);
    cv::Mat get(int layer) const;
    int getNumberOfLayers() const;
private:
    std::vector<cv::Mat> layers_;
    int number_of_layers;
};
