#include <vector>
#include <opencv2/opencv.hpp>

class gauss_pyr
{
public:
    gauss_pyr(cv::Mat& img, int _number_of_layers, float sigma);
    cv::Mat get(int layer) const;
    int getNumberOfLayers() const;
private:
    std::vector<cv::Mat> layers_;
    int number_of_layers;
};
