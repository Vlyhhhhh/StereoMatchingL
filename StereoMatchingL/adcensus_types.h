/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
*			  https://github.com/ethan-li-coding/AD-Census
* Describe	: header of adcensus_types
*/

#ifndef ADCENSUS_STEREO_TYPES_H_
#define ADCENSUS_STEREO_TYPES_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <opencv2/highgui.hpp>   //opencv����
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>

using std::vector;
using std::pair;
using namespace cv;

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

/** \brief �������ͱ��� */
typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

/** \brief float32��Чֵ */
constexpr auto Invalid_Float = std::numeric_limits<float32>::infinity();

constexpr auto Large_Float = 99999.0f;
constexpr auto Small_Float = -99999.0f;

/** \brief Census���ڳߴ����� */
enum CensusSize {
	Census5x5 = 0,
	Census9x7
};

/** \brief ADCensus�����ṹ�� */
struct SILCStereoOption {
	sint32  min_disparity;		// ��С�Ӳ�
	sint32	max_disparity;		// ����Ӳ�

	sint32	lambda_ad;			// ����AD����ֵ�Ĳ���
	sint32	lambda_census;		// ����Census����ֵ�Ĳ���
	sint32  lambda_sobel;       // Parameters that control the Sobel cost value
	sint32	cross_L1;			// ʮ�ֽ��洰�ڵĿռ��������L1
	sint32  cross_L2;			// ʮ�ֽ��洰�ڵĿռ��������L2
	sint32	cross_t1;			// ʮ�ֽ��洰�ڵ���ɫ�������t1
	sint32  cross_t2;			// ʮ�ֽ��洰�ڵ���ɫ�������t2
	float32	so_p1;				// ɨ�����Ż�����p1
	float32	so_p2;				// ɨ�����Ż�����p2
	sint32	so_tso;				// ɨ�����Ż�����tso
	sint32	irv_ts;				// Iterative Region Voting������ts
	float32 irv_th;				// Iterative Region Voting������th

	float32	lrcheck_thres;		// ����һ����Լ����ֵ

	bool	do_lr_check;					// �Ƿ�������һ����
	bool	do_filling;						// �Ƿ����Ӳ����
	bool	do_discontinuity_adjustment;	// �Ƿ���������������

	SILCStereoOption() : min_disparity(0), max_disparity(64),
		lambda_ad(10), lambda_census(30), lambda_sobel(100),
		cross_L1(34), cross_L2(17),
		cross_t1(20), cross_t2(6),
		so_p1(1.0f), so_p2(3.0f),
		so_tso(15), irv_ts(20), irv_th(0.4f),
		lrcheck_thres(1.0f),
		do_lr_check(true), do_filling(true), do_discontinuity_adjustment(false) {};
};

/**
* \brief ��ɫ�ṹ��
*/
struct ADColor {
	uint8 r, g, b;
	ADColor() : r(0), g(0), b(0) {}
	ADColor(uint8 _b, uint8 _g, uint8 _r) {
		r = _r; g = _g; b = _b;
	}
};

#endif