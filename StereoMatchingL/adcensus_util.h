/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of adcensus_util
*/

#pragma once
#include <algorithm>
#include "adcensus_types.h"
struct center//定义超像素中心点
{
	int x;//column
	int y;//row
	int L;
	int A;
	int B;
	int label;
};
// 定义超像素结构体
struct SuperPixel {
	Point center; // 中心点坐标
	Rect region;  // 区域
};

namespace adcensus_util
{
	/**
	* \brief census变换
	* \param source	输入，影像数据
	* \param census	输出，census值数组，预分配空间
	* \param width	输入，影像宽
	* \param height	输入，影像高
	*/
	void census_transform_9x7(const uint8* source, vector<uint64>& census, const sint32& width, const sint32& height);
	// Hamming距离
	uint8 Hamming64(const uint64& x, const uint64& y);

	/**
	* \brief 中值滤波
	* \param in				输入，源数据
	* \param out			输出，目标数据
	* \param width			输入，宽度
	* \param height			输入，高度
	* \param wnd_size		输入，窗口宽度
	*/
	void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);

	/**
	* \brief sobel计算梯度
	* \param source			输入，灰度数据
	* \param sobel			输出，sobel梯度数组
	* \param kernel_x			输入，宽度
	* \param kernel_y			输入，高度
	*/
	void Sobel(const uint8* source, vector<uint8>&sobel, const sint32& width, const sint32& height, Mat& kernel_x, Mat&  kernel_y);

	/**
	* \brief sobel计算8方向梯度
	* \param source			输入，灰度数据
	* \param sobel			输出，sobel梯度数组
	* \param kernel_x			输入，宽度
	* \param kernel_y			输入，高度
	*/
	void Sobel_8(const uint8* source, vector<uint8>&sobel, const sint32& width, const sint32& height, float sobelFilters[8][25]);

	/**Liu added
	*\brief Superpixel segmentation initilizeCenters initializes cluster centers
	* \param imageLAB		input，LABimage
	* \param centers		output，the vectors of all center points
	* \param len		    input，Side length of the grid
	*/
	int initilizeCenters(cv::Mat &imageLAB, std::vector<center> &centers, int len);

	/**Liu added
	*\brief  Superpixel segmentation fituneCenter moves the cluster center to the place with the smallest gradient in the surrounding 8 neighborhoods, and the gradient is calculated using Sobel
	* \param imageLAB		input，LABimage
	* \param sobelGradient	input，Sobel gradient map
	* \param centers		input，Vectors of all center points
	*/
	int fituneCenter(cv::Mat &imageLAB, cv::Mat &sobelGradient, std::vector<center> &centers);

	/**Liu added
	*\brief  Superpixel segmentation clustering
	* \param imageLAB		input，LABimage
	* \param DisMask	    input，Distance Mask
	* \param labelMask		input，LabelMask
	* \param centers		input，Vectors of all center points
	* \param len		    input，Search scope when clustering
	* \param m		        input，Balance coefficient between spatial distance and color distance
	*/
	int clustering(const cv::Mat &imageLAB, cv::Mat &DisMask, cv::Mat &labelMask, std::vector<center> &centers, int len, int m);

	/**Liu added
	*\brief  Superpixel segmentation updateCenter updates the cluster center
	* \param imageLAB		input，LABimage
	* \param centers		input，the vectors of all center points
	* \param len		    input，Search scope when clustering
	*/
	int updateCenter(cv::Mat &imageLAB, cv::Mat &labelMask, std::vector<center> &centers, int len);

	/** Liu added
	*\brief  Superpixel segmentation showSLICResult displays clustering results
	* \param image		    input，BGRimage
	* \param labelMask		input，LabelMask
	* \param centers		input，Vectors of all center points
	* \param len		    input，Search scope when clustering
	* \param dst		    output，SLICResult
	*/
	int showSLICResult(const cv::Mat &image, cv::Mat &labelMask, std::vector<center> &centers, int len, cv::Mat &dst);

	/**Liu added
	*\brief  SLIC
	*/
	int SLIC(cv::Mat &image, cv::Mat &resultLabel, std::vector<center> &centers, int len, int m);

	/**Liu added
	*\brief  mat to vector
	*/
	void matToVector(const cv::Mat& mat, std::vector<uint8_t>& vec);

}