/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: implement of class CostComputor
*/

#include "cost_computor.h"
#include "adcensus_util.h"

CostComputor::CostComputor() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
lambda_ad_(0), lambda_census_(0), min_disparity_(0), max_disparity_(0),
is_initialized_(false), slic_left_(), slic_right_() {}

CostComputor::~CostComputor()
{

}

bool CostComputor::Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;

	const sint32 img_size = width_ * height_;
	const sint32 disp_range = max_disparity_ - min_disparity_;
	if (img_size <= 0 || disp_range <= 0) {
		is_initialized_ = false;
		return false;
	}

	// 灰度数据（左右影像）
	gray_left_.resize(img_size);
	gray_right_.resize(img_size);
	// census数据（左右影像）
	census_left_.resize(img_size, 0);
	census_right_.resize(img_size, 0);
	// Liu:sobel data（left image and right image）
	sobel_left_.resize(img_size, 0);
	sobel_right_.resize(img_size, 0);


	// 初始代价数据
	cost_init_.resize(img_size * disp_range);

	is_initialized_ = !gray_left_.empty() && !gray_right_.empty() && !census_left_.empty() && !census_right_.empty() && !cost_init_.empty();
	return is_initialized_;
}

void CostComputor::SetData(const uint8* img_left, const uint8* img_right)
{
	img_left_ = img_left;
	img_right_ = img_right;
}

void CostComputor::SetParams(const sint32& lambda_ad, const sint32& lambda_census, const sint32& lambda_sobel)
{
	lambda_ad_ = lambda_ad;
	lambda_census_ = lambda_census;
	lambda_sobel_ = lambda_sobel;
}

void CostComputor::ComputeGray()
{
	// 彩色转灰度
	for (sint32 n = 0; n < 2; n++) {
		const auto color = (n == 0) ? img_left_ : img_right_;
		auto& gray = (n == 0) ? gray_left_ : gray_right_;
		for (sint32 y = 0; y < height_; y++) {
			for (sint32 x = 0; x < width_; x++) {
				const auto b = color[y * width_ * 3 + 3 * x];
				const auto g = color[y * width_ * 3 + 3 * x + 1];
				const auto r = color[y * width_ * 3 + 3 * x + 2];
				gray[y * width_ + x] = uint8(r * 0.299 + g * 0.587 + b * 0.114);
			}
		}
	}
}

void CostComputor::CensusTransform()
{
	// 左右影像census变换
	adcensus_util::census_transform_9x7(&gray_left_[0], census_left_, width_, height_);
	adcensus_util::census_transform_9x7(&gray_right_[0], census_right_, width_, height_);
}

//Liu:
void CostComputor::Calculatesobel()
{
	//Calculate Sobel gradient
	float values_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	Mat kernel_x = Mat_<float>(3, 3, values_x);

	float values_y[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	Mat kernel_y = Mat_<float>(3, 3, values_y);

	//Calculate image gradient (the gradient of the guide image and the gradient of the original image)
	adcensus_util::Sobel(&gray_left_[0], sobel_left_, width_, height_, kernel_x, kernel_y);
	adcensus_util::Sobel(&gray_right_[0], sobel_right_, width_, height_, kernel_x, kernel_y);

}

//Liu:Calculate 8-direction sobel
void CostComputor::Calculatesobel_8()
{
	float sobelFilters[8][25] = {
		// 0 degrees
		{0, 0, 0, 0, 0,
		 -1, -2, -4, -2, -1,
		 0, 0, 0, 0, 0,
		 1, 2, 4, 2, 1,
		 0, 0, 0, 0, 0},
		 // 22.5 degrees
		 {0, 0, 0, 0, 0,
		  0, -2, -4, -2, 0,
		  -1, -4, 0, 4, 1,
		  0, 2, 4, 2, 0,
		  0, 0, 0, 0, 0},
		  // 45 degrees
		  {0, 0, 0, -1, 0,
		   0, -2, -4, 0, 1,
		   0, -4, 0, 4, 0,
		   -1, 0, 4, 2, 0,
		   0, 1, 0, 0, 0},
		   // 67.5 degrees
		   {0, 0, -1, 0, 0,
			0, -2, -4, 2, 0,
			0, -4, 0, 4, 0,
			0, -2, 4, 2, 0,
			0, 0, 1, 0, 0},
			// 90 degrees
			{0, -1, 0, 1, 0,
			 0, -2, 0, 2, 0,
			 0, -4, 0, 4, 0,
			 0, -2, 0, 2, 0,
			 0, -1, 0, 1, 0},
			 // 112.5 degrees
			 {0, 0, 1, 0, 0,
			  0, -2, 4, 2, 0,
			  0, -4, 0, 4, 0,
			  0, -2, -4, 2, 0,
			  0, 0, -1, 0, 0},
			  // 135 degrees
			  {0, 1, 0, 0, 0,
			   -1, 0, 4, 2, 0,
			   0, -4, 0, 4, 0,
			   0, -2, -4, 0, 1,
			   0, 0, 0, -1, 0},
			   // 157.5 degrees
			   {0, 0, 0, 0, 0,
				0, 2, 4, 2, 0,
				-1, -4, 0,4, 1,
				0, -2, -4, -2, 0,
				0, 0, 0, 0, 0}
	};

	//Calculate image gradient (the gradient of the guide image and the gradient of the original image)
	adcensus_util::Sobel_8(&gray_left_[0], sobel_left_8, width_, height_, sobelFilters);
	adcensus_util::Sobel_8(&gray_right_[0], sobel_right_8, width_, height_, sobelFilters);

}
//Liu
void CostComputor::Calculateslic(cv::Mat& img_left, cv::Mat& img_right)
{
	int row = height_;
	int col = width_;

	cv::Mat labelMask; // save every pixel's label
	cv::Mat dst;       // save the shortest distance to the nearest centers
	std::vector<center> centers; // clustering centers

	int len = 12; // the scale of the superpixel ,len*len；
	int m = 30;   // a parameter which adjusts the weights of spatial distance and the color space distance,

	// SLIC superpixel segmentation
	adcensus_util::SLIC(img_left, labelMask, centers, len, m);
	adcensus_util::SLIC(img_right, labelMask, centers, len, m);

	// Display SLIC segmentation results
	adcensus_util::showSLICResult(img_left, labelMask, centers, len, slic_left_);
	adcensus_util::showSLICResult(img_right, labelMask, centers, len, slic_right_);

	// Convert the result to vector
	adcensus_util::matToVector(slic_left_, slic_left);
	adcensus_util::matToVector(slic_right_, slic_right);
}

void CostComputor::ComputeCost()
{
	const sint32 disp_range = max_disparity_ - min_disparity_;

	// 预设参数
	const auto lambda_ad = lambda_ad_;
	const auto lambda_census = lambda_census_;
	const auto lambda_sobel = lambda_sobel_;
	// 计算代价
	for (sint32 y = 0; y < height_; y++) {
		for (sint32 x = 0; x < width_; x++) {
			const auto bl = img_left_[y * width_ * 3 + 3 * x];//先计算好左边的RGB
			const auto gl = img_left_[y * width_ * 3 + 3 * x + 1];
			const auto rl = img_left_[y * width_ * 3 + 3 * x + 2];
			const auto& census_val_l = census_left_[y * width_ + x];//左边censuss
			const auto& sobel_val_l = sobel_left_[y * width_ + x];//左边sobel
			const auto& sobel_val_l_8 = sobel_left_8[y * width_ + x];//左边sobel-8方向
			const auto& slic_val_l = slic_left[y * width_ + x];//左边超像素值

			// 逐视差计算代价值
			for (sint32 d = min_disparity_; d < max_disparity_; d++) {
				auto& cost = cost_init_[y * width_ * disp_range + x * disp_range + (d - min_disparity_)];
				const sint32 xr = x - d;
				if (xr < 0 || xr >= width_) {
					cost = 1.0f;
					continue;
				}

				// ad代价
				const auto br = img_right_[y * width_ * 3 + 3 * xr];//在计算右边的RGB
				const auto gr = img_right_[y * width_ * 3 + 3 * xr + 1];
				const auto rr = img_right_[y * width_ * 3 + 3 * xr + 2];
				const float32 cost_ad = (abs(bl - br) + abs(gl - gr) + abs(rl - rr)) / 3.0f;

				// census代价
				const auto& census_val_r = census_right_[y * width_ + xr];
				const float32 cost_census = static_cast<float32>(adcensus_util::Hamming64(census_val_l, census_val_r));

				//Liu:sobel cost
				const auto sobel_val_r = sobel_right_[y * width_ + x];
				const float32 cost_sobel = static_cast<float32>((sobel_val_l - sobel_val_r) / 6.0f);

				//Liu:sobel cost,8 directions
				const auto sobel_val_r_8 = sobel_right_8[y * width_ + x];
				const float32 cost_sobel_8 = static_cast<float32>((sobel_val_l_8 - sobel_val_r_8) / 6.0f);

				//Liu:
				const auto slic_val_r = slic_right[y * width_ + x];

				// ad-census代价
			   /* cost = 1 - exp(-cost_ad / lambda_ad) + 1 - exp(-cost_census / lambda_census);*/

			   //Liu：
				if (slic_val_l != slic_val_r) {
					// Located at the superpixel boundary
					cost = 1 - exp(-cost_ad / lambda_ad) + 1 - exp(-cost_census / lambda_census) + 1 - exp(-cost_sobel / 100);
				}
				else {
					cost = 1 - exp(-cost_ad / lambda_ad) + 1 - exp(-cost_census / lambda_census) + 1 - exp(-cost_sobel_8 / 150);
				}

			}
		}
	}
}

void CostComputor::Compute()
{
	if (!is_initialized_) {
		return;
	}

	// 将 uint8* 类型转换为 cv::Mat
	cv::Mat img_left_mat(height_, width_, CV_8UC3, const_cast<uint8*>(img_left_));
	cv::Mat img_right_mat(height_, width_, CV_8UC3, const_cast<uint8*>(img_right_));


	// 计算灰度图
	ComputeGray();

	// census变换
	CensusTransform();

	//Liu:Sobel calculates the gradient
	Calculatesobel();

	//Liu:Sobel calculates the gradient-8 directions
	Calculatesobel_8();

	//Liu:SLIC
	Calculateslic(img_left_mat, img_right_mat);

	// 代价计算
	ComputeCost();
}

float32* CostComputor::get_cost_ptr()
{
	if (!cost_init_.empty()) {
		return &cost_init_[0];
	}
	else {
		return nullptr;
	}
}
