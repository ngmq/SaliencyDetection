#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "oriented_pyr.h"
#include <string>

bool computeBinaryMap( cv::InputArray _saliencyMap, cv::OutputArray _binaryMap )
{
  cv::Mat saliencyMap = _saliencyMap.getMat();
  cv::Mat labels = cv::Mat::zeros( saliencyMap.rows * saliencyMap.cols, 1, 1 );
  cv::Mat samples = cv::Mat_<float>( saliencyMap.rows * saliencyMap.cols, 1 );
  cv::Mat centers;
  cv::TermCriteria terminationCriteria;
  terminationCriteria.epsilon = 0.1;
  terminationCriteria.maxCount = 5000;
  terminationCriteria.type = cv::TermCriteria::COUNT + cv::TermCriteria::EPS;

  int elemCounter = 0;
  for ( int i = 0; i < saliencyMap.rows; i++ )
  {
    for ( int j = 0; j < saliencyMap.cols; j++ )
    {
      samples.at<float>( elemCounter, 0 ) = saliencyMap.at<float>( i, j );
      elemCounter++;
    }
  }

  cv::kmeans( samples, 5, labels, terminationCriteria, 5, cv::KMEANS_RANDOM_CENTERS, centers );

  cv::Mat outputMat = cv::Mat_<float>( saliencyMap.size() );
  int intCounter = 0;
  for ( int x = 0; x < saliencyMap.rows; x++ )
  {
    for ( int y = 0; y < saliencyMap.cols; y++ )
    {
      outputMat.at<float>( x, y ) = centers.at<float>( labels.at<int>( intCounter, 0 ), 0 );
      intCounter++;
    }

  }

  //Convert
  outputMat = outputMat * 255;
  outputMat.convertTo( outputMat, CV_8U );

  // adaptative thresholding using Otsu's method, to make saliency map binary
  _binaryMap.createSameSize(outputMat, outputMat.type());
  cv::Mat BinaryMap = _binaryMap.getMat();
  cv::threshold( outputMat, BinaryMap, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU );

  return true;

}

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

double minVal, maxVal;

int main(int argc, char** argv )
{
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  
    gray.convertTo(gray, CV_32F);
    gray /= 255.0f;
    
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
    int nLayers = 3;
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
        
        
        cv::minMaxLoc(layer, &minVal, &maxVal);
        std::cout << "Minval = " << minVal << std::endl;
        
        cv::threshold(layer, layer, maxVal * 0.6, 1, cv::THRESH_BINARY);
        //cv::normalize(layer, layer2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        //cv::imshow("Across Scale", layer);
        //cv::waitKey(0);
        
        if( firstAddition )
        {
        		mOrientation = cv::Mat(layer.size(), CV_32F, 0.0);
        		firstAddition = false;
        }
        mOrientation += layer;
    }
    mOrientation /= num_orientations;
    
    //std::cout << mOrientation << std::endl;
    /*cv::namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
    cv::imshow("Orientation", mOrientation);
    cv::waitKey(0);*/

	 /*cv::Mat mFinal = mOrientation.clone();   
	 cv::minMaxLoc(mFinal, &minVal, &maxVal); 
    cv::threshold(mFinal, mFinal, maxVal * 0.6, 1, cv::THRESH_BINARY);
    
    
    cv::namedWindow("Final", CV_WINDOW_AUTOSIZE);
	 cv::imshow("Final", mFinal);
	 cv::waitKey(0);*/
    
    cv::FileStorage fs("myfile.yaml", cv::FileStorage::WRITE );
	 fs << "Orientation" << mOrientation;
	 fs.release();
	 
	 /***
	 Gaussian and Laplacian Pyramid
	 ***/
	 
	 float center_sigma = 2.0;
    float surround_sigma = 16.0;

	  gauss_pyr c_pyr_l(lab_channels[0], nLayers, center_sigma);
    gauss_pyr c_pyr_a(lab_channels[1], nLayers, center_sigma);
    gauss_pyr c_pyr_b(lab_channels[2], nLayers, center_sigma);
    gauss_pyr s_pyr_l(lab_channels[0], nLayers, surround_sigma);
    gauss_pyr s_pyr_a(lab_channels[1], nLayers, surround_sigma);
    gauss_pyr s_pyr_b(lab_channels[2], nLayers, surround_sigma);
    
	cv::Mat FCS, FSC, mF_RG, mF_BY, mF_Intensity;

    std::vector<cv::Mat> CS_vec;
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_a.get(i);
        cv::Mat surround = s_pyr_a.get(i);
        
        cv::Mat CS = center - surround;
        cv::threshold(CS, CS, 0, 1, cv::THRESH_TOZERO);
        CS_vec.push_back(CS);
    }
    FCS = across_scale_addition(CS_vec);
	 //cv::namedWindow("Feature map CS", CV_WINDOW_AUTOSIZE);
    //cv::imshow("Feature map CS", FCS );
    //cv::waitKey(0);
    
    std::vector<cv::Mat> SC_vec;
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_a.get(i);
        cv::Mat surround = s_pyr_a.get(i);
        
        cv::Mat SC = -center + surround;
        cv::threshold(SC, SC, 0, 1, cv::THRESH_TOZERO);
        SC_vec.push_back(SC);
    }
    
    FSC = across_scale_addition(SC_vec);
	 //cv::namedWindow("Feature map SC", CV_WINDOW_AUTOSIZE);
    //cv::imshow("Feature map SC", FSC );
    //cv::waitKey(0);
    
    mF_RG = 0.5 * (FCS + FSC);
    //cv::Mat Color_RG_F = F.clone();
/*    cv::namedWindow("Feature map a", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map a", mF_RG);
    cv::waitKey(0);*/
    
    // binarize
/*    cv::minMaxLoc(F, &minVal, &maxVal);
	 cv::normalize(F, F, 0, maxVal, cv::NORM_MINMAX, CV_32F);
	 cv::threshold(F, F, maxVal * 0.4, 1, CV_8UC1);
	 
	 cv::namedWindow("Red/Green", CV_WINDOW_AUTOSIZE);
	 cv::imshow("Red/Green", F);
	 cv::waitKey(0);*/
	 
	 /***
	 Blue Yellow
	 ***/
	 
	 CS_vec.clear();
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_b.get(i);
        cv::Mat surround = s_pyr_b.get(i);
        
        cv::Mat CS = center - surround;
        cv::threshold(CS, CS, 0, 1, cv::THRESH_TOZERO);
        CS_vec.push_back(CS);
    }
    FCS = across_scale_addition(CS_vec);
	 /*cv::namedWindow("Feature map CS b", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map CS b", FCS );
    cv::waitKey(0);*/
    
    SC_vec.clear();
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_b.get(i);
        cv::Mat surround = s_pyr_b.get(i);
        
        cv::Mat SC = -center + surround;
        cv::threshold(SC, SC, 0, 1, cv::THRESH_TOZERO);
        SC_vec.push_back(SC);
    }
    
    FSC = across_scale_addition(SC_vec);
	 /*cv::namedWindow("Feature map SC b", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map SC b", FSC );
    cv::waitKey(0);*/
    
    mF_BY = 0.5 * (FCS + FSC);
   /* cv::namedWindow("Feature map b", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map b", mF_BY);
    cv::waitKey(0);*/
	 
	 /*** Intensity
	 ***/
	 
	 CS_vec.clear();
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_l.get(i);
        cv::Mat surround = s_pyr_l.get(i);
        
        cv::Mat CS = center - surround;
        cv::threshold(CS, CS, 0, 1, cv::THRESH_TOZERO);
        CS_vec.push_back(CS);
    }
    FCS = across_scale_addition(CS_vec);
	 /*cv::namedWindow("Feature map CS l", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map CS l", FCS );
    cv::waitKey(0);*/
    
    SC_vec.clear();
    for (int i = 0; i < nLayers; ++i)
    {
        cv::Mat center = c_pyr_l.get(i);
        cv::Mat surround = s_pyr_l.get(i);
        
        cv::Mat SC = -center + surround;
        cv::threshold(SC, SC, 0, 1, cv::THRESH_TOZERO);
        SC_vec.push_back(SC);
    }
    
    FSC = across_scale_addition(SC_vec);
	 /*cv::namedWindow("Feature map SC l", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map SC l", FSC );
    cv::waitKey(0);*/
    
    mF_Intensity = 0.5 * (FCS + FSC);
   /* cv::namedWindow("Feature map l", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map l", mF_Intensity );
    cv::waitKey(0);*/
    
	 /*** DONE ***/
	 
	 
	 double max_m0, max_F_RG, max_F_BY, max_F_Intensity, max_final;
	 cv::minMaxLoc(mOrientation, &minVal, &max_m0);
	 cv::minMaxLoc(mF_RG, &minVal, &max_F_RG);
	 cv::minMaxLoc(mF_BY, &minVal, &max_F_BY);
	 cv::minMaxLoc(mF_Intensity, &minVal, &max_F_Intensity);
	 
    max_final = cv::max(max_m0, max_F_RG);
    max_final = cv::max(max_final, max_F_BY);
    max_final = cv::max(max_final, max_F_Intensity);
    
    //std::cout << max_m0 << "; " << max_F_RG << "; " << max_final << std::endl;
    cv::normalize(mOrientation, mOrientation, 0, max_final, cv::NORM_MINMAX, CV_32F);
	 cv::normalize(mF_RG, mF_RG, 0, max_final, cv::NORM_MINMAX, CV_32F);
	 cv::normalize(mF_BY, mF_BY, 0, max_final, cv::NORM_MINMAX, CV_32F);
	 cv::normalize(mF_Intensity, mF_Intensity, 0, max_final, cv::NORM_MINMAX, CV_32F);
	 
	 cv::namedWindow("Orientation", CV_WINDOW_AUTOSIZE);
    cv::imshow("Orientation", mOrientation);
    cv::waitKey(0);
    
    cv::namedWindow("Feature map l", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map l", mF_Intensity );
    cv::waitKey(0);
    
    cv::namedWindow("Feature map a", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map a", mF_RG);
    cv::waitKey(0);
    
 	  cv::namedWindow("Feature map b", CV_WINDOW_AUTOSIZE);
    cv::imshow("Feature map b", mF_BY);
    cv::waitKey(0);
	 
	 cv::Mat mFinal = mOrientation + mF_RG + mF_BY + mF_Intensity;
	 cv::minMaxLoc(mFinal, &minVal, &max_final);
	 
	 
	 cv::Mat layer = mFinal.clone();		 
	 cv::normalize(layer, layer, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	 cv::namedWindow("Final2", CV_WINDOW_AUTOSIZE);
	 cv::imshow("Final2", layer);
	 cv::waitKey(0);
	 
	 cv::threshold(mFinal, mFinal, max_final * 0.6, 1, CV_8UC1);
	 
	 cv::namedWindow("Final", CV_WINDOW_AUTOSIZE);
	 cv::imshow("Final", mFinal);
	 cv::waitKey(0);
    
    return 0;
}
