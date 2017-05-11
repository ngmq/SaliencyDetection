#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "oriented_pyr.h"

cv::Mat across_scale_addition(const std::vector<cv::Mat>& scale_images)
{
    cv::Size im_size = scale_images[0].size();
    cv::Mat result = scale_images[0];
    for (int i = 1; i < scale_images.size(); ++i)
    {
        cv::Mat resized;
        if ( scale_images[i].size() != im_size )
        {
            cv::resize(scale_images[i], resized, im_size, 0, 0, cv::INTER_CUBIC);
        }
        else
        {
            resized = scale_images[0];
        }

        result += resized;
    }
    return result;
}

int main(int argc, char** argv )
{
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
  
    gray.convertTo(gray, CV_32F);
    //gray /= 255.0f;
    
    //std::cout << gray << std::endl;
	
	 cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    // Normalize to range [0, 1] and convert to CV_32F for calculations
    lab.convertTo(lab, CV_32F);
    lab /= 255.0f;

    // split the channels (L, a, b) to vector elements
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    //cv::namedWindow("Intensity", CV_WINDOW_NORMAL);
    //cv::imshow("Intensity", lab_channels[0]);
    //cv::waitKey(0);

    //cv::namedWindow("Gauss Pyramid", CV_WINDOW_AUTOSIZE);
    int nLayers = 4;
    double sigma = 2.0;
    gauss_pyr gp(gray, nLayers, sigma);
    for( int i = 0; i < nLayers; ++i )
    {
    	  //cv::Mat layer = gp.get(i);
    	  //cv::normalize(layer, layer, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //cv::imshow("Gauss Pyramid", layer);
        //cv::waitKey(0);
    }

    //cv::namedWindow("Laplacian Pyramid", CV_WINDOW_AUTOSIZE);
    double sigma2 = 1.0;
    laplacian_pyr lp(gp, sigma2);
    for( int i = 0; i < nLayers; ++i )
    {
    	  //cv::Mat layer = lp.get(i);
    	  //cv::normalize(layer, layer, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    	  
        //cv::imshow("Laplacian Pyramid", layer);        
        //cv::waitKey(0);
    }

    int num_orientations = 8;
    
    oriented_pyr op(lp, num_orientations);
    //cv::namedWindow("Oriented Pyramid", CV_WINDOW_AUTOSIZE);
    for( int i = 0; i < num_orientations; ++i )
    {
        for( int j = 0; j < nLayers; ++j )
        {
        		//cv::Mat layer = op.get(i, j);
        		//cv::normalize(layer, layer, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            //cv::imshow("Oriented Pyramid", layer);
            //cv::waitKey(0);
        }        
    }

	 std::cout << "Across Scale displaying..." << std::endl;
	 
	 //cv::namedWindow("Across Scale", CV_WINDOW_AUTOSIZE);
    cv::Mat mOrientation;
    bool firstAddition = true;    
    for(int i = 0; i < num_orientations; ++i)
    {
        cv::Mat layer = across_scale_addition(op.getByOrientation(i));
        
        //cv::Mat layer2;
        //cv::normalize(layer, layer2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //cv::imshow("Across Scale", layer2);
        //cv::waitKey(0);
        if( firstAddition )
        {
        		mOrientation = cv::Mat(layer.size(), CV_32F, 0.0);
        		firstAddition = false;
        }
        mOrientation += layer;
    }
    
    //std::cout << mOrientation << std::endl;
    cv::Mat mOrientation2;
    cv::namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
    cv::normalize(mOrientation, mOrientation2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("Orientation", mOrientation2);
    cv::waitKey(0);
    
    cv::FileStorage fs("myfile.yml", cv::FileStorage::WRITE );
	 fs << "Orientation" << mOrientation;
	 fs.release();
    
    return 0;
}
