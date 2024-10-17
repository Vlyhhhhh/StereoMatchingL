/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of adcensus_util
*/

#pragma once
#include <algorithm>
#include "adcensus_types.h"
struct center//���峬�������ĵ�
{
	int x;//column
	int y;//row
	int L;
	int A;
	int B;
	int label;
};
// ���峬���ؽṹ��
struct SuperPixel {
	Point center; // ���ĵ�����
	Rect region;  // ����
};

namespace adcensus_util
{
	/**
	* \brief census�任
	* \param source	���룬Ӱ������
	* \param census	�����censusֵ���飬Ԥ����ռ�
	* \param width	���룬Ӱ���
	* \param height	���룬Ӱ���
	*/
	void census_transform_9x7(const uint8* source, vector<uint64>& census, const sint32& width, const sint32& height);
	// Hamming����
	uint8 Hamming64(const uint64& x, const uint64& y);

	/**
	* \brief ��ֵ�˲�
	* \param in				���룬Դ����
	* \param out			�����Ŀ������
	* \param width			���룬���
	* \param height			���룬�߶�
	* \param wnd_size		���룬���ڿ��
	*/
	void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);

	/**
	* \brief sobel�����ݶ�
	* \param source			���룬�Ҷ�����
	* \param sobel			�����sobel�ݶ�����
	* \param kernel_x			���룬���
	* \param kernel_y			���룬�߶�
	*/
	void Sobel(const uint8* source, vector<uint8>&sobel, const sint32& width, const sint32& height, Mat& kernel_x, Mat&  kernel_y);

	/**
	* \brief sobel����8�����ݶ�
	* \param source			���룬�Ҷ�����
	* \param sobel			�����sobel�ݶ�����
	* \param kernel_x			���룬���
	* \param kernel_y			���룬�߶�
	*/
	void Sobel_8(const uint8* source, vector<uint8>&sobel, const sint32& width, const sint32& height, float sobelFilters[8][25]);

	/**Liu added
	*\brief Superpixel segmentation initilizeCenters initializes cluster centers
	* \param imageLAB		input��LABimage
	* \param centers		output��the vectors of all center points
	* \param len		    input��Side length of the grid
	*/
	int initilizeCenters(cv::Mat &imageLAB, std::vector<center> &centers, int len);

	/**Liu added
	*\brief  Superpixel segmentation fituneCenter moves the cluster center to the place with the smallest gradient in the surrounding 8 neighborhoods, and the gradient is calculated using Sobel
	* \param imageLAB		input��LABimage
	* \param sobelGradient	input��Sobel gradient map
	* \param centers		input��Vectors of all center points
	*/
	int fituneCenter(cv::Mat &imageLAB, cv::Mat &sobelGradient, std::vector<center> &centers);

	/**Liu added
	*\brief  Superpixel segmentation clustering
	* \param imageLAB		input��LABimage
	* \param DisMask	    input��Distance Mask
	* \param labelMask		input��LabelMask
	* \param centers		input��Vectors of all center points
	* \param len		    input��Search scope when clustering
	* \param m		        input��Balance coefficient between spatial distance and color distance
	*/
	int clustering(const cv::Mat &imageLAB, cv::Mat &DisMask, cv::Mat &labelMask, std::vector<center> &centers, int len, int m);

	/**Liu added
	*\brief  Superpixel segmentation updateCenter updates the cluster center
	* \param imageLAB		input��LABimage
	* \param centers		input��the vectors of all center points
	* \param len		    input��Search scope when clustering
	*/
	int updateCenter(cv::Mat &imageLAB, cv::Mat &labelMask, std::vector<center> &centers, int len);

	/** Liu added
	*\brief  Superpixel segmentation showSLICResult displays clustering results
	* \param image		    input��BGRimage
	* \param labelMask		input��LabelMask
	* \param centers		input��Vectors of all center points
	* \param len		    input��Search scope when clustering
	* \param dst		    output��SLICResult
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