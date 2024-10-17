/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: main
*/
#include <iostream>
#include "SILCStereo.h"
#include <chrono>
using namespace std::chrono;

// opencv library
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world343d.lib")
#else
#pragma comment(lib,"opencv_world343.lib")
#endif

/*��ʾ�Ӳ�ͼ*/
void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name);
/*�����Ӳ�ͼ*/
void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/*����float32�Ӳ�ͼ*/
void SaveDisparityMap32(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path);
/**
* \brief
* \param argv 3
* \param argc argc[1]:��Ӱ��·�� argc[2]: ��Ӱ��·�� argc[3]: ��С�Ӳ�[��ѡ��Ĭ��0] argc[4]: ����Ӳ�[��ѡ��Ĭ��64]
* \param eg. ..\Data\trainingQ\Adirondack\im0.png ..\Data\trainingQ\Adirondack\im1.png 0 64
* \param eg. ..\Data\Cloth3\view1.png ..\Data\Cloth3\view5.png 0 128
* \return
*/
int main(int argv, char** argc)
{
	if (argv < 3) {
		std::cout << "�������٣�������ָ������Ӱ��·����" << std::endl;
		return -1;
	}

	printf("Image Loading...");
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ȡӰ��
	std::string path_left = argc[1];
	std::string path_right = argc[2];

	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);

	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
		return -1;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
		return -1;
	}


	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// ����Ӱ��Ĳ�ɫ����
	auto bytes_left = new uint8[width * height * 3];
	auto bytes_right = new uint8[width * height * 3];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}
	printf("Done!\n");

	// AD-Censusƥ��������
	SILCStereoOption ad_option;
	// ��ѡ�ӲΧ
	ad_option.min_disparity = argv < 4 ? 0 : atoi(argc[3]);
	ad_option.max_disparity = argv < 5 ? 64 : atoi(argc[4]);
	// һ���Լ����ֵ
	ad_option.lrcheck_thres = 1.0f;

	// �Ƿ�ִ��һ���Լ��
	ad_option.do_lr_check = true;

	// �Ƿ�ִ���Ӳ����
	// �Ӳ�ͼ���Ľ�������ɿ��������̣���������䣬�����У�������
	ad_option.do_filling = true;

	printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, ad_option.min_disparity, ad_option.max_disparity);

	// ����SILCƥ����ʵ�� Liu:2024.5
	SILCStereo SILC_matching;

	printf("AD-Census Initializing...\n");
	auto start = steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʼ��
	if (!SILC_matching.Initialize(width, height, ad_option)) {
		std::cout << "AD-Census��ʼ��ʧ�ܣ�" << std::endl;
		return -2;
	}
	auto end = steady_clock::now();
	auto tt = duration_cast<milliseconds>(end - start);
	printf("AD-Census Initializing Done! Timing :	%lf s\n\n", tt.count() / 1000.0);

	printf("AD-Census Matching...\n");
	// disparity���鱣�������ص��Ӳ���
	auto disparity = new float32[uint32(width * height)]();

	start = steady_clock::now();
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ƥ��
	if (!SILC_matching.Match(bytes_left, bytes_right, disparity)) {
		std::cout << "AD-Censusƥ��ʧ�ܣ�" << std::endl;
		return -2;
	}
	end = steady_clock::now();
	tt = duration_cast<milliseconds>(end - start);
	printf("\nAD-Census Matching...Done! Timing :	%lf s\n", tt.count() / 1000.0);

	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ʾ�Ӳ�ͼ
	ShowDisparityMap(disparity, width, height, "disp-left");
	// �����Ӳ�ͼ
	SaveDisparityMap(disparity, width, height, path_left);
	// Liu modified and saved it as float32 type disparity map
	SaveDisparityMap32(disparity, width, height, path_left);

	cv::waitKey(0);

	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// �ͷ��ڴ�
	delete[] disparity;
	disparity = nullptr;
	delete[] bytes_left;
	bytes_left = nullptr;
	delete[] bytes_right;
	bytes_right = nullptr;

	system("pause");
	return 0;
}

void ShowDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& name)
{
	// ��ʾ�Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(name, disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imshow(name + "-color", disp_color);

}

void SaveDisparityMap(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// �����Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
	float32 min_disp = float32(width), max_disp = -float32(width);
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = abs(disp_map[i * width + j]);
			if (disp == Invalid_Float) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imwrite(path + "-d.png", disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imwrite(path + "-c.png", disp_color);
}
void SaveDisparityMap32(const float32* disp_map, const sint32& width, const sint32& height, const std::string& path)
{
	// �����Ӳ�ͼ
	cv::Mat disp_mat = cv::Mat(height, width, CV_32FC1);
	float32 min_disp = std::numeric_limits<float32>::max();
	float32 max_disp = std::numeric_limits<float32>::min();

	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			const float32 disp = std::abs(disp_map[i * width + j]);
			if (disp != Invalid_Float) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
			disp_mat.at<float32>(i, j) = disp_map[i * width + j];
		}
	}


	cv::imwrite(path + "-32d.png", disp_mat);
}
