#pragma once
/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of class CostComputor
*/

#ifndef SILCSTEREO_COST_COMPUTOR_H_
#define SILCSTEREO_COST_COMPUTOR_H_

#include "adcensus_types.h"

/**
 * \brief ���ۼ�������
 */
class CostComputor {
public:
	CostComputor();
	~CostComputor();

	/**
	 * \brief ��ʼ��
	 * \param width			Ӱ���
	 * \param height		Ӱ���
	 * \param min_disparity	��С�Ӳ�
	 * \param max_disparity	����Ӳ�
	 * \return true: ��ʼ���ɹ�
	 */
	bool Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity);

	/**
	 * \brief ���ô��ۼ�����������
	 * \param img_left		// ��Ӱ�����ݣ���ͨ��
	 * \param img_right		// ��Ӱ�����ݣ���ͨ��
	 */
	void SetData(const uint8* img_left, const uint8* img_right);

	/**
	 * \brief ���ô��ۼ������Ĳ���
	 * \param lambda_ad		// lambda_ad
	 * \param lambda_census // lambda_census
	  * \param lambda_sobel // lambda_sobel
	 */
	void SetParams(const sint32& lambda_ad, const sint32& lambda_census, const sint32& lambda_sobel);

	/** \brief �����ʼ���� */
	void Compute();

	/** \brief ��ȡ��ʼ��������ָ�� */
	float32* get_cost_ptr();

private:
	/** \brief ����Ҷ����� */
	void ComputeGray();

	/** \brief Census�任 */
	void CensusTransform();

	/** \brief Sobel extracts gradients */
	void Calculatesobel();

	/** \brief Sobel extracts 8-directional gradients */
	void Calculatesobel_8();

	/** \brief silc */
	//void Calculateslic();
	void  Calculateslic(cv::Mat& img_left, cv::Mat& img_right);

	/** \brief ������� */
	void ComputeCost();
private:
	/** \brief ͼ��ߴ� */
	sint32	width_;
	sint32	height_;

	/** \brief Ӱ������ */
	const uint8* img_left_;
	const uint8* img_right_;

	/** \brief ��Ӱ��Ҷ�����	 */
	vector<uint8> gray_left_;
	/** \brief ��Ӱ��Ҷ�����	 */
	vector<uint8> gray_right_;

	/** \brief ��Ӱ��census����	*/
	vector<uint64> census_left_;
	/** \brief ��Ӱ��census����	*/
	vector<uint64> census_right_;

	/** \brief ��ʼƥ�����	*/
	vector<float32> cost_init_;

	/** \brief lambda_ad*/
	sint32 lambda_ad_;
	/** \brief lambda_census*/
	sint32 lambda_census_;
	/** \brief lambda_sobel*/
	sint32 lambda_sobel_;

	/** \brief ��С�Ӳ�ֵ */
	sint32 min_disparity_;
	/** \brief ����Ӳ�ֵ */
	sint32 max_disparity_;

	/** \brief Sobelx convolution kernel	*/
	Mat kernel_x;
	/** \brief Sobely convolution kernel	*/
	Mat kernel_y;

	/** \brief Left image sobel array	*/
	vector<uint8> sobel_left_;
	/** \brief Right image sobel array	*/
	vector<uint8> sobel_right_;

	/** \brief Left image 8-sobel array	*/
	vector<uint8> sobel_left_8;
	/** \brief Right image 8-sobel array	*/
	vector<uint8> sobel_right_8;

	/** \brief  Left image superpixel segmentation array	*/
	vector<uint8> slic_left;
	/** \brief  Right image superpixel segmentation array	*/
	vector<uint8> slic_right;

	/** \brief Left image superpixel segmentation array	*/
	Mat  slic_left_;
	/** \brief Right image superpixel segmentation array	*/
	Mat  slic_right_;

	/** \brief �Ƿ�ɹ���ʼ����־	*/
	bool is_initialized_;
};
#endif