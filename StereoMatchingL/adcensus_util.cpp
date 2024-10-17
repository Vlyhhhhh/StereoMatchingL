/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: implement of adcensus_util
*/

#include "adcensus_util.h"
#include <cassert>

void adcensus_util::census_transform_9x7(const uint8* source, vector<uint64>& census, const sint32& width, const sint32& height)
{
	if (source == nullptr || census.empty() || width <= 9 || height <= 7) {
		return;
	}

	// 逐像素计算census值
	for (sint32 i = 4; i < height - 4; i++) {
		for (sint32 j = 3; j < width - 3; j++) {

			// 中心像素值
			const uint8 gray_center = source[i * width + j];

			// 遍历大小为9x7的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
			uint64 census_val = 0u;
			for (sint32 r = -4; r <= 4; r++) {
				for (sint32 c = -3; c <= 3; c++) {
					census_val <<= 1;
					const uint8 gray = source[(i + r) * width + j + c];
					if (gray < gray_center) {
						census_val += 1;
					}
				}
			}

			// 中心像素的census值
			census[i * width + j] = census_val;
		}
	}
}


uint8 adcensus_util::Hamming64(const uint64& x, const uint64& y)
{
	uint64 dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val) {
		++dist;
		val &= val - 1;
	}

	return static_cast<uint8>(dist);
}

void adcensus_util::MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size)
{
	const sint32 radius = wnd_size / 2;
	const sint32 size = wnd_size * wnd_size;

	std::vector<float32> wnd_data;
	wnd_data.reserve(size);

	for (sint32 y = 0; y < height; y++) {
		for (sint32 x = 0; x < width; x++) {
			wnd_data.clear();
			for (sint32 r = -radius; r <= radius; r++) {
				for (sint32 c = -radius; c <= radius; c++) {
					const sint32 row = y + r;
					const sint32 col = x + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}
			std::sort(wnd_data.begin(), wnd_data.end());
			if (!wnd_data.empty()) {
				out[y * width + x] = wnd_data[wnd_data.size() / 2];
			}
		}
	}

}
void adcensus_util::Sobel(const uint8* source, vector<uint8>&sobel, const sint32& width, const sint32& height, Mat& kernel_x, Mat&  kernel_y)
{


	int height_x = kernel_x.rows;
	int width_x = kernel_x.cols;

	int height_y = kernel_y.rows;
	int width_y = kernel_y.cols;

	for (int row = 1; row < height - 1; row++)
	{
		for (int col = 1; col < width - 1; col++)
		{
			float G_X = 0;
			for (int h = 0; h < height_x; h++)
			{
				for (int w = 0; w < width_x; w++)
				{
					G_X += source[row + h - 1, col + w - 1] * kernel_x.at<float>(h, w);
				}
			}

			float G_Y = 0;
			for (int h = 0; h < height_y; h++)
			{
				for (int w = 0; w < width_y; w++)
				{
					G_Y += source[row + h - 1, col + w - 1] * kernel_y.at<float>(h, w);
				}
			}

			sobel[row* width + col] = saturate_cast<uchar>(cv::abs(G_X) + cv::abs(G_Y));//[i * width + j]
			//output_img.at<uchar>(row, col) = saturate_cast<uchar>(cv::abs(G_Y));

		}
	}

}
void adcensus_util::Sobel_8(const uint8* source, vector<uint8>&sobel_8, const sint32& width, const sint32& height, float sobelFilters[8][25])
{
	sobel_8.resize(width * height);

	for (int i = 2; i < height - 2; ++i) {
		for (int j = 2; j < width - 2; ++j) {
			int maxVal = 0;
			for (int k = 0; k < 8; ++k) {
				int sum = 0;
				for (int m = -2; m <= 2; ++m) {
					for (int n = -2; n <= 2; ++n) {
						sum += source[(i + m) * width + (j + n)] * sobelFilters[k][(m + 2) * 5 + (n + 2)];
					}
				}
				if (sum > maxVal) {
					maxVal = sum;
				}
			}
			sobel_8[i * width + j] = maxVal;
		}
	}
}

//Liu added generate the initial center point
int adcensus_util::initilizeCenters(cv::Mat &imageLAB, std::vector<center> &centers, int len)
{
	if (imageLAB.empty())
	{
		std::cout << "In itilizeCenters:     image is empty!\n";
		return -1;
	}

	uchar *ptr = NULL;
	center cent;
	int num = 0;
	for (int i = 0; i < imageLAB.rows; i += len)
	{
		cent.y = i + len / 2;
		if (cent.y >= imageLAB.rows) continue;
		ptr = imageLAB.ptr<uchar>(cent.y);
		for (int j = 0; j < imageLAB.cols; j += len)
		{
			cent.x = j + len / 2;
			if ((cent.x >= imageLAB.cols)) continue;
			cent.L = *(ptr + cent.x * 3);
			cent.A = *(ptr + cent.x * 3 + 1);
			cent.B = *(ptr + cent.x * 3 + 2);
			cent.label = ++num;
			centers.push_back(cent);
		}
	}
	return 0;
}
int adcensus_util::fituneCenter(cv::Mat &imageLAB, cv::Mat &sobelGradient, std::vector<center> &centers)
{
	if (sobelGradient.empty()) return -1;

	center cent;
	double *sobPtr = sobelGradient.ptr<double>(0);
	uchar *imgPtr = imageLAB.ptr<uchar>(0);
	int w = sobelGradient.cols;
	for (int ck = 0; ck < centers.size(); ck++)
	{
		cent = centers[ck];
		if (cent.x - 1 < 0 || cent.x + 1 >= sobelGradient.cols || cent.y - 1 < 0 || cent.y + 1 >= sobelGradient.rows)
		{
			continue;
		}//end if
		double minGradient = 9999999;
		int tempx = 0, tempy = 0;
		for (int m = -1; m < 2; m++)
		{
			sobPtr = sobelGradient.ptr<double>(cent.y + m);
			for (int n = -1; n < 2; n++)
			{
				double gradient = pow(*(sobPtr + (cent.x + n) * 3), 2)
					+ pow(*(sobPtr + (cent.x + n) * 3 + 1), 2)
					+ pow(*(sobPtr + (cent.x + n) * 3 + 2), 2);
				if (gradient < minGradient)
				{
					minGradient = gradient;
					tempy = m;//row
					tempx = n;//column
				}//end if
			}
		}
		cent.x += tempx;
		cent.y += tempy;
		imgPtr = imageLAB.ptr<uchar>(cent.y);
		centers[ck].x = cent.x;
		centers[ck].y = cent.y;
		centers[ck].L = *(imgPtr + cent.x * 3);
		centers[ck].A = *(imgPtr + cent.x * 3 + 1);
		centers[ck].B = *(imgPtr + cent.x * 3 + 2);

	}//end for
	return 0;
}
int adcensus_util::clustering(const cv::Mat &imageLAB, cv::Mat &DisMask, cv::Mat &labelMask, std::vector<center> &centers, int len, int m)
{
	if (imageLAB.empty())
	{
		std::cout << "clustering :the input image is empty!\n";
		return -1;
	}

	double *disPtr = NULL;//disMask type: 64FC1
	double *labelPtr = NULL;//labelMask type: 64FC1
	const uchar *imgPtr = NULL;//imageLAB type: 8UC3

	//disc = std::sqrt(pow(L - cL, 2)+pow(A - cA, 2)+pow(B - cB,2))
	//diss = std::sqrt(pow(x-cx,2) + pow(y-cy,2));
	//dis = sqrt(disc^2 + (diss/len)^2 * m^2)
	double dis = 0, disc = 0, diss = 0;
	//cluster center's cx, cy,cL,cA,cB;
	int cx, cy, cL, cA, cB, clabel;
	//imageLAB's  x, y, L,A,B
	int x, y, L, A, B;

	for (int ck = 0; ck < centers.size(); ++ck)
	{
		cx = centers[ck].x;
		cy = centers[ck].y;
		cL = centers[ck].L;
		cA = centers[ck].A;
		cB = centers[ck].B;
		clabel = centers[ck].label;

		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 | i >= imageLAB.rows) continue;
			//pointer point to the ith row
			imgPtr = imageLAB.ptr<uchar>(i);
			disPtr = DisMask.ptr<double>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 | j >= imageLAB.cols) continue;
				L = *(imgPtr + j * 3);
				A = *(imgPtr + j * 3 + 1);
				B = *(imgPtr + j * 3 + 2);

				disc = std::sqrt(pow(L - cL, 2) + pow(A - cA, 2) + pow(B - cB, 2));
				diss = std::sqrt(pow(j - cx, 2) + pow(i - cy, 2));
				dis = sqrt(pow(disc, 2) + m * pow(diss, 2));

				if (dis < *(disPtr + j))
				{
					*(disPtr + j) = dis;
					*(labelPtr + j) = clabel;
				}//end if
			}//end for
		}
	}//end for (int ck = 0; ck < centers.size(); ++ck)


	return 0;
}



int  adcensus_util::updateCenter(cv::Mat &imageLAB, cv::Mat &labelMask, std::vector<center> &centers, int len)
{
	double *labelPtr = NULL;//labelMask type: 64FC1
	const uchar *imgPtr = NULL;//imageLAB type: 8UC3
	int cx, cy;

	for (int ck = 0; ck < centers.size(); ++ck)
	{
		double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0;
		cx = centers[ck].x;
		cy = centers[ck].y;
		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 | i >= imageLAB.rows) continue;
			//pointer point to the ith row
			imgPtr = imageLAB.ptr<uchar>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 | j >= imageLAB.cols) continue;

				if (*(labelPtr + j) == centers[ck].label)
				{
					sumL += *(imgPtr + j * 3);
					sumA += *(imgPtr + j * 3 + 1);
					sumB += *(imgPtr + j * 3 + 2);
					sumx += j;
					sumy += i;
					sumNum += 1;
				}//end if
			}
		}
		//update center
		if (sumNum == 0) sumNum = 0.000000001;
		centers[ck].x = sumx / sumNum;
		centers[ck].y = sumy / sumNum;
		centers[ck].L = sumL / sumNum;
		centers[ck].A = sumA / sumNum;
		centers[ck].B = sumB / sumNum;

	}//end for

	return 0;
}
int adcensus_util::showSLICResult(const cv::Mat &image, cv::Mat &labelMask, std::vector<center> &centers, int len, cv::Mat &out)
{
	cv::Mat dst = image.clone();
	cv::cvtColor(dst, dst, cv::COLOR_BGR2Lab);
	double *labelPtr = NULL;//labelMask type: 32FC1
	uchar *imgPtr = NULL;//image type: 8UC3

	int cx, cy;
	double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0.00000001;
	for (int ck = 0; ck < centers.size(); ++ck)
	{
		cx = centers[ck].x;
		cy = centers[ck].y;

		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 | i >= image.rows) continue;
			//pointer point to the ith row
			imgPtr = dst.ptr<uchar>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 | j >= image.cols) continue;

				if (*(labelPtr + j) == centers[ck].label)
				{
					*(imgPtr + j * 3) = centers[ck].L;
					*(imgPtr + j * 3 + 1) = centers[ck].A;
					*(imgPtr + j * 3 + 2) = centers[ck].B;
				}//end if
			}
		}
	}//end for

	cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
	out = dst;
	cv::namedWindow("showSLIC", 0);
	cv::imshow("showSLIC", dst);

	return 0;
}
int adcensus_util::SLIC(cv::Mat &image, cv::Mat &resultLabel, std::vector<center> &centers, int len, int m)
{
	if (image.empty())
	{
		std::cout << "in SLIC the input image is empty!\n";
		return -1;

	}

	int MAXDIS = 999999;
	int height, width;
	height = image.rows;
	width = image.cols;

	//convert color
	cv::Mat imageLAB;
	cv::cvtColor(image, imageLAB, cv::COLOR_BGR2Lab);

	//get sobel gradient map
	cv::Mat sobelImagex, sobelImagey, sobelGradient;
	cv::Sobel(imageLAB, sobelImagex, CV_64F, 0, 1, 3);
	cv::Sobel(imageLAB, sobelImagey, CV_64F, 1, 0, 3);
	cv::addWeighted(sobelImagex, 0.5, sobelImagey, 0.5, 0, sobelGradient);//sobel output image type is CV_64F

	//initiate
	//std::vector<center> centers;
	//disMask save the distance of the pixels to center;
	cv::Mat disMask;
	//labelMask save the label of the pixels
	cv::Mat labelMask = cv::Mat::zeros(cv::Size(width, height), CV_64FC1);

	//initialize centers,  get centers
	initilizeCenters(imageLAB, centers, len);
	//if the center locates in the edges, fitune it's location
	fituneCenter(imageLAB, sobelGradient, centers);

	//update cluster 10 times 
	for (int time = 0; time < 10; time++)
	{
		//clustering
		disMask = cv::Mat(height, width, CV_64FC1, cv::Scalar(MAXDIS));
		clustering(imageLAB, disMask, labelMask, centers, len, m);
		//update
		updateCenter(imageLAB, labelMask, centers, len);
		//fituneCenter(imageLAB, sobelGradient, centers);
	}

	resultLabel = labelMask;

	return 0;
}

void adcensus_util::matToVector(const cv::Mat& mat, std::vector<uint8_t>& vec)
{
	// 确保vec大小足够容纳mat的像素值
	vec.resize(mat.rows * mat.cols);

	// 遍历mat的像素值，并存放到vec中
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			vec[i * mat.cols + j] = mat.at<uint8_t>(i, j);
		}
	}
}
