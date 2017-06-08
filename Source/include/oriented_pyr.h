#include <vector>
#include <opencv2/opencv.hpp>
#include "laplacian_pyr.h"

class oriented_pyr
{
public:
    oriented_pyr(const laplacian_pyr& p, int num_orientations, int size, int lamda);
    std::vector <cv::Mat > getByLayer(int layer) const;
    std::vector <cv::Mat > getByOrientation( int orientation) const;
    cv::Mat get(int orientation, int layer) const;

private:
    int number_of_layers;
    int number_of_orientations;
    std::vector <std::vector <cv::Mat > > orientation_maps;

};
