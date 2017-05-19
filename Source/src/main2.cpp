#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "oriented_pyr.h"
#include <string>

void show(std::string name, const cv::Mat &m)
{
  cv::imshow(name, m);
  cv::waitKey(0);
}

// add maps in pyramid
cv::Mat acrossScaleAddition(const std::vector<cv::Mat>& scale_images)
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

// count the number of local maxima (uniqueness)
int countLocalMaxima(const cv::Mat& input, const int kernel_size)
{
  cv::Mat src = input.clone();
	cv::normalize(src, src, 0, 255, cv::NORM_MINMAX, CV_8U);
	src.convertTo(src, CV_8U);

	int cnt = 0;

	for(int i = 0; i < src.rows; ++i)
	{
		for(int j = 0; j < src.cols; ++j)
		{
			unsigned char center = src.at<unsigned char>(i, j);
			if(center <= 100)
				continue;
			bool isLocal = true;
			for(int dx = -1; dx <= 1; ++dx)
			{
				for(int dy = -1; dy <= 1; ++dy)
				{
					unsigned char surround = src.at<unsigned char>(i + dx, j + dy);
					if( (dx != 0 || dy != 0) && surround > center )
					{
						isLocal = false;
					}
				}
			}
			cnt += isLocal;
		}
	}
	if(cnt == 1)
	{
		std::cout << src << std::endl;
	}
	return cnt;
}

// find the maximum of normalization
double normalizationFactor(const std::vector<cv::Mat>& m)
{
  // normalization factor to use
  double min, max, temp_min, temp_max, desired_max;
  cv::minMaxLoc(m[0].clone(), &min, &max);
  for(int i=1; i<m.size(); ++i)
  {
    cv::minMaxLoc(m[i].clone(), &temp_min, &temp_max);
    desired_max = std::max( max, temp_max);
  }
  std::cout << "desired max " << desired_max << std::endl;
  return desired_max;
}

// normalize all conspicuity map
std::vector<cv::Mat> normalizeVectors(const std::vector<cv::Mat>& m)
{
  double normalization_factor = normalizationFactor(m);
  for(int i=0; i<m.size();++i)
  {
    cv::normalize(m[i], m[i], 0.0f, (float)normalization_factor, cv::NORM_MINMAX);
  }
  return m;
}

// arithmetic mean operation fusion
cv::Mat meanOperation(const std::vector<cv::Mat>& m)
{
  double normalization_factor = normalizationFactor(m);
  cv::Mat conspicuity_map = cv::Mat(m[0].size(), CV_32F, 0.0);

  for(int i=0; i<m.size();++i)
  {
    cv::Mat layer = m[i].clone();
    conspicuity_map += layer;
  }
  conspicuity_map /= (float)m.size();
  cv::normalize(conspicuity_map, conspicuity_map, 0.0f, (float)normalization_factor, cv::NORM_MINMAX);

  return conspicuity_map;
}

// max operation fusion
cv::Mat maxOperation(const std::vector<cv::Mat>& m)
{
  double normalization_factor = normalizationFactor(m);
  cv::Mat conspicuity_map = m[0].clone();
  for(int i=1; i<m.size();++i)
  {
    conspicuity_map = cv::max(conspicuity_map, m[i].clone());
  }
  cv::normalize(conspicuity_map, conspicuity_map, 0.0f, (float)normalization_factor, cv::NORM_MINMAX);

  return conspicuity_map;
}

// uniqueness weighting fusion
cv::Mat uniqueness(std::vector<cv::Mat>& v)
{
  float sum = 0.0f;
  cv::Mat final = cv::Mat(v[0].size(), CV_32F, 0.0);;

  for(int i=0; i<v.size(); ++i)
  {
    int n_local_m = countLocalMaxima(v[i], 3);
    float w = n_local_m == 0 ? 0 : sqrt(1.0 / n_local_m);
    sum += w;
    final += w*v[i];
  }
  final /= sum;
  return final;
}

//on-off
cv::Mat onOffMap(const gauss_pyr &on, const gauss_pyr &off, int num_layers)
{
  std::vector<cv::Mat> cs_vec;
  for (int i = 0; i < num_layers; ++i)
    {
        cv::Mat center = on.get(i);
        cv::Mat surround = off.get(i);

        cv::Mat cs = center - surround;
        cv::threshold(cs, cs, 0, 1, cv::THRESH_TOZERO);
        cs_vec.push_back(cs.clone());
    }
  cv::Mat on_off = acrossScaleAddition(cs_vec);
  return on_off;
}

//off_on
cv::Mat offOnMap(const gauss_pyr &on, const gauss_pyr &off, int num_layers)
{
  std::vector<cv::Mat> sc_vec;
  for (int i = 0; i < num_layers; ++i)
    {
        cv::Mat center = on.get(i);
        cv::Mat surround = off.get(i);

        cv::Mat sc = -center + surround;
        cv::threshold(sc, sc, 0, 1, cv::THRESH_TOZERO);
        sc_vec.push_back(sc.clone());
    }
  cv::Mat off_on = acrossScaleAddition(sc_vec);
  return off_on;
}

// create a vector of color conspicuity maps
std::vector<cv::Mat> colorConspicuityMaps(std::vector<cv::Mat> &lab_channels, int num_layers, int type=0)
{
  cv::Mat L_conspicuity_map, A_conspicuity_map, B_conspicuity_map;
  std::vector<cv::Mat> L_vector, A_vector, B_vector, v;
  float center_sigma = 2.0;
  float surround_sigma = 16.0;

  gauss_pyr c_pyr_l(lab_channels[0], num_layers, center_sigma);
  gauss_pyr c_pyr_a(lab_channels[1], num_layers, center_sigma);
  gauss_pyr c_pyr_b(lab_channels[2], num_layers, center_sigma);
  gauss_pyr s_pyr_l(lab_channels[0], num_layers, surround_sigma);
  gauss_pyr s_pyr_a(lab_channels[1], num_layers, surround_sigma);
  gauss_pyr s_pyr_b(lab_channels[2], num_layers, surround_sigma);

  cv::Mat on_off = onOffMap(c_pyr_l, s_pyr_l, num_layers);
  cv::Mat off_on = onOffMap(c_pyr_l, s_pyr_l, num_layers);
  L_vector.push_back(on_off.clone());
  L_vector.push_back(off_on.clone());

  on_off = onOffMap(c_pyr_a, s_pyr_a, num_layers);
  off_on = onOffMap(c_pyr_a, s_pyr_a, num_layers);
  A_vector.push_back(on_off.clone());
  A_vector.push_back(off_on.clone());

  on_off = onOffMap(c_pyr_b, s_pyr_b, num_layers);
  off_on = onOffMap(c_pyr_b, s_pyr_b, num_layers);
  B_vector.push_back(on_off.clone());
  B_vector.push_back(off_on.clone());

  if(type==0)
  {
    L_conspicuity_map = meanOperation(L_vector);
    A_conspicuity_map = meanOperation(A_vector);
    B_conspicuity_map = meanOperation(B_vector);
  }
  else if (type==1)
  {
    L_conspicuity_map = maxOperation(L_vector);
    A_conspicuity_map = maxOperation(A_vector);
    B_conspicuity_map = maxOperation(B_vector);
  }
  else
  {
    L_conspicuity_map = uniqueness(L_vector);
    A_conspicuity_map = uniqueness(A_vector);
    B_conspicuity_map = uniqueness(B_vector);
  }

  /*cv::Mat display;
  cv::normalize(L_conspicuity_map, display, 0, 1, cv::NORM_MINMAX);
  cv::namedWindow("L", CV_WINDOW_NORMAL);
  show("L", display);

  cv::normalize(A_conspicuity_map, display, 0, 1, cv::NORM_MINMAX);
  cv::namedWindow("A", CV_WINDOW_NORMAL);
  show("A", display);

  cv::normalize(B_conspicuity_map, display, 0, 1, cv::NORM_MINMAX);
  cv::namedWindow("B", CV_WINDOW_NORMAL);
  show("B", display);*/

  v.push_back(L_conspicuity_map.clone());
  v.push_back(A_conspicuity_map.clone());
  v.push_back(B_conspicuity_map.clone());

  return v;
}

// create a orientation feature map
std::vector<cv::Mat> orientationFeatureMap(cv::Mat &m, int num_layers, int num_orientations)
{
  double minVal, maxVal;
  const char *title[] = {"1", "2", "3", "4", "5", "6", "7", "8"};

  double sigma = 2.0;
  gauss_pyr gp(m, num_layers, sigma);

  double sigma2 = 1.0;
  laplacian_pyr lp(gp, sigma2);

  oriented_pyr op(lp, num_orientations);

  std::vector<cv::Mat> orientation_maps;
  for(int i = 0; i < num_orientations; ++i)
  {
    cv::Mat layer = acrossScaleAddition(op.getByOrientation(i));
    cv::minMaxLoc(layer, &minVal, &maxVal);
    cv::threshold(layer, layer, maxVal * 0.6, 1, cv::THRESH_BINARY);
    orientation_maps.push_back(layer.clone());

    /*cv::Mat display;
    cv::normalize(layer, display, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow(title[i], CV_WINDOW_NORMAL);
    show(title[i], display);*/
  }

  return orientation_maps;
}

// create a orientation conspicuity map
cv::Mat orientationConspicuityMap(const cv::Mat &m, int num_layers, int num_orientations, int type=0)
{
  cv::Mat img = m.clone();
  std::vector<cv::Mat> feature_maps = orientationFeatureMap(img, num_layers, num_orientations);
  cv::Mat conspicuity_map;
  if(type == 0)
  {
    conspicuity_map = meanOperation(feature_maps);
  }
  else if(type == 1)
  {
    conspicuity_map = maxOperation(feature_maps);
  }
  else
  {
    conspicuity_map = uniqueness(feature_maps);
  }

  /*cv::Mat display;
  cv::normalize(conspicuity_map, display, 0, 1, cv::NORM_MINMAX);
  cv::namedWindow("Orienation final", CV_WINDOW_NORMAL);
  show("Orientation final", display);*/
  return conspicuity_map;
}

int main(int argc, char** argv )
{
  const std::string PREFIX = "../output/";
  const std::string EXTENSION = ".png";
  std::string path = argv[1];
  std::vector<cv::String> fn;
  cv::glob(path,fn,true); // recurse
  std::size_t tesa = fn[0].find("*");
  for (size_t k=0; k<fn.size(); ++k)
  {
    std::string temp = fn[k];
    std::size_t pos1 = 1;
    while(pos1 != tesa)
    {
       pos1 = temp.find("/");
       temp = temp.substr(pos1+1);
    }
    std::size_t pos2 = temp.find(".jpg");
    std::string name = temp.substr(pos1+1,pos2-pos1-1);
    //std::cout << name << std::endl;

    cv::Mat image = cv::imread(fn[k], CV_LOAD_IMAGE_COLOR);
    if (image.empty()) continue;

    // gray image
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    gray /= 255.0f;

    // lab image
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    // Normalize to range [0, 1] and convert to CV_32F for calculations
    lab.convertTo(lab, CV_32F);
    lab /= 255.0f;

    // split the channels (L, a, b) to vector elements
    std::vector<cv::Mat> lab_channels;
    cv::split(lab, lab_channels);

    int num_layers = 4;
    int num_orientations = 8;

    // mean operation
    /*std::vector<cv::Mat> conspicuity_maps = colorConspicuityMaps(lab_channels, num_layers);
    conspicuity_maps.push_back(orientationConspicuityMap(gray, num_layers, num_orientations));
    conspicuity_maps = normalizeVectors(conspicuity_maps);
    cv::Mat final = meanOperation(conspicuity_maps);
    cv::Mat display;
    cv::normalize(final, display, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow("Final1", CV_WINDOW_NORMAL);
    show("Final1", display);*/

    // max operation
    /*conspicuity_maps = colorConspicuityMaps(lab_channels, num_layers, 1);
    conspicuity_maps.push_back(orientationConspicuityMap(gray, num_layers, num_orientations, 1));
    final = maxOperation(conspicuity_maps);
    cv::normalize(final, display, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow("Final2", CV_WINDOW_NORMAL);
    show("Final2", display);*/

    // unique weighting
    std::vector<cv::Mat> conspicuity_maps = colorConspicuityMaps(lab_channels, num_layers, 2);
    conspicuity_maps.push_back(orientationConspicuityMap(gray, num_layers, num_orientations, 2));
    cv::Mat final = uniqueness(conspicuity_maps);
    //final.convertTo(final, CV_8U);
    cv::normalize(final, final, 0, 255, cv::NORM_MINMAX);
    //cv::namedWindow("Final3", CV_WINDOW_NORMAL);
    //show("Final3", display);

    cv::imwrite(PREFIX+name+EXTENSION, final);
  }

  return 0;
}
