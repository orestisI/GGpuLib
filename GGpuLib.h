#ifndef GGPULIB_H
#define GGPULIB_H

#include<stdlib.h>
#include<stdio.h>
#include<iostream>
#include<cstdarg>
#include<cuda_runtime.h>
#include<math.h>
#include<curand.h>
#include<time.h>
#include<string.h>


//GpuGeneral Functions
template<class T>
__global__ void GpuGeneralCpy(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = src[i];
	}
}

template<class T>
__global__ void GpuGeneralInit(T *res,int resLength,int resBach,T val){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = val;
	}
}

template<class T>
__global__ void GpuGeneralLinearInit(T *res,int resLength,int resBach){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = i;
	}
}

//-General Multiplication
template<class T>
__global__ void GpuGeneralMul(T *res,int resLength,int resBach,int resColumns,
		int commonDim,T *arr1,T *arr2){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	int row,column,ofst;
	T sum = 0;

	for (int i=from; i<to; i++){
		row = i / resColumns;
		column = i % resColumns;

		ofst = commonDim * row;
		for (int k=0; k<commonDim; k++){
			sum = sum + arr1[ofst + k] * arr2[resColumns * k + column];
		}
		res[i] = sum;
	}
}

//Assumes same length
template<class T>
__global__ void GpuGeneralAdd(T *res,int resLength,int resBach,T *arr1,T *arr2){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] + arr2[i];
	}
}

template<class T>
__global__ void GpuGeneralAddCnst(T *res,int resLength,int resBach,T *arr1,T val){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] + val;
	}
}

//Broadcasting arr2
template<class T>
__global__ void GpuGeneralAddBcast(T *res,int resLength,int resBach,T *arr1,
	int *arr1Dim,int arr1DimLength,int *arr1Ofst,T *arr2,
	int *arr2Dim,int arr2DimLength,int *arr2Ofst){

	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int ofst = arr1DimLength - arr2DimLength;
	int linearPos;
	int xDim;
	for (int i=from; i<to; i++){
		int temp = i;
		linearPos = 0;
		if (ofst >= 0){
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				if (k >= ofst){
					linearPos = linearPos + (xDim % arr2Dim[k - ofst])*arr2Ofst[k - ofst + 1];
				}
				temp = temp % arr1Ofst[k + 1];
			}
		}
		else{
			ofst = - ofst;
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				linearPos = linearPos + (xDim % arr2Dim[k + ofst])*arr2Ofst[k + ofst + 1];
				temp = temp % arr1Ofst[k + 1];
			}
		}
		res[i] = arr1[i] + arr2[linearPos];
	}
}

//Assumes same length
template<class T>
__global__ void GpuGeneralSub(T *res,int resLength,int resBach,T *arr1,T *arr2){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] - arr2[i];
	}
}

template<class T>
__global__ void GpuGeneralSubCnst(T *res,int resLength,int resBach,T *arr1,T val){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] - val;
	}
}
//Broadcasting arr2
template<class T>
__global__ void GpuGeneralSubBcast(T *res,int resLength,int resBach,T *arr1,
	int *arr1Dim,int arr1DimLength,int *arr1Ofst,T *arr2,
	int *arr2Dim,int arr2DimLength,int *arr2Ofst){

	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int ofst = arr1DimLength - arr2DimLength;
	int linearPos;
	int xDim;
	for (int i=from; i<to; i++){
		int temp = i;
		linearPos = 0;
		if (ofst >= 0){
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				if (k >= ofst){
					linearPos = linearPos + (xDim % arr2Dim[k - ofst])*arr2Ofst[k - ofst + 1];
				}
				temp = temp % arr1Ofst[k + 1];
			}
		}
		else{
			ofst = - ofst;
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				linearPos = linearPos + (xDim % arr2Dim[k + ofst])*arr2Ofst[k + ofst + 1];
				temp = temp % arr1Ofst[k + 1];
			}
		}
		res[i] = arr1[i] - arr2[linearPos];
	}
}
//Assumes same length
template<class T>
__global__ void GpuGeneralDot(T *res,int resLength,int resBach,T *arr1,T *arr2){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] * arr2[i];
	}
}

template<class T>
__global__ void GpuGeneralDotCnst(T *res,int resLength,int resBach,T *arr1,T val){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] * val;
	}
}
//Broadcasting arr2
template<class T>
__global__ void GpuGeneralDotBcast(T *res,int resLength,int resBach,T *arr1,
	int *arr1Dim,int arr1DimLength,int *arr1Ofst,T *arr2,
	int *arr2Dim,int arr2DimLength,int *arr2Ofst){

	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int ofst = arr1DimLength - arr2DimLength;
	int linearPos;
	int xDim;
	for (int i=from; i<to; i++){
		int temp = i;
		linearPos = 0;
		if (ofst >= 0){
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				if (k >= ofst){
					linearPos = linearPos + (xDim % arr2Dim[k - ofst])*arr2Ofst[k - ofst + 1];
				}
				temp = temp % arr1Ofst[k + 1];
			}
		}
		else{
			ofst = - ofst;
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				linearPos = linearPos + (xDim % arr2Dim[k + ofst])*arr2Ofst[k + ofst + 1];
				temp = temp % arr1Ofst[k + 1];
			}
		}
		res[i] = arr1[i] * arr2[linearPos];
	}
}

template<class T>
__global__ void GpuGeneralDiv(T *res,int resLength,int resBach,T *arr1,T *arr2){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] / arr2[i];
	}
}

template<class T>
__global__ void GpuGeneralDivCnst(T *res,int resLength,int resBach,T *arr1,T val){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = arr1[i] / val;
	}
}
//Broadcasting arr2
template<class T>
__global__ void GpuGeneralDivBcast(T *res,int resLength,int resBach,T *arr1,
	int *arr1Dim,int arr1DimLength,int *arr1Ofst,T *arr2,
	int *arr2Dim,int arr2DimLength,int *arr2Ofst){

	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int ofst = arr1DimLength - arr2DimLength;
	int linearPos;
	int xDim;
	for (int i=from; i<to; i++){
		int temp = i;
		linearPos = 0;
		if (ofst >= 0){
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				if (k >= ofst){
					linearPos = linearPos + (xDim % arr2Dim[k - ofst])*arr2Ofst[k - ofst + 1];
				}
				temp = temp % arr1Ofst[k + 1];
			}
		}
		else{
			ofst = - ofst;
			for (int k=0; k<arr1DimLength; k++){
				xDim = temp / arr1Ofst[k + 1];
				linearPos = linearPos + (xDim % arr2Dim[k + ofst])*arr2Ofst[k + ofst + 1];
				temp = temp % arr1Ofst[k + 1];
			}
		}
		res[i] = arr1[i] / arr2[linearPos];
	}
}



//Assumes 1 block is running and arr1 arr2 has the same length
template<class T>
__global__ void GpuGeneralEqual(int *res,T *arr1,int arr1Length,int bach,T *arr2){
	int from = threadIdx.x * bach;
	int to = from + bach;

	if (to > arr1Length){
		to = arr1Length;
	}
	int temp = 1;
	int i = from;
	while(temp && (i < to)){
		temp = (arr1[i] == arr2[i]);
		i++;
	}
	atomicAnd(res,temp);
}

template<class T>
__global__ void GpuGeneralSum(T *res,T *arr1,int arr1Length){
	T sum = 0;
	for (int i=0; i<arr1Length; i++){
		sum = sum + arr1[i];
	}
	*res = sum;
}

template<class T>
__global__ void GpuGeneralSigmoid(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = 1.0/(1.0 + exp(-src[i]));
	}
}

template<class T>
__global__ void GpuGeneralLog(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		res[i] = log(src[i]);
	}
}

template<class T>
__global__ void GpuGeneralReverse(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		res[i] = src[resLength - 1 - i];
	}
	
}

template<class T>
__global__ void GpuGeneralTranspose(T *res,int resLength,int resBach,int *resOfst,T *arr1,
	int arr1DimLength,int *arr1Ofst){

	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int linearPos;
	int xDim;
	for (int i=from; i<to; i++){
		int temp = i;
		linearPos = 0;
		for (int k=0; k<arr1DimLength; k++){
			xDim = temp / resOfst[k + 1];
			linearPos = linearPos + xDim * arr1Ofst[arr1DimLength - k];
			temp = temp % resOfst[k + 1];
		}
		res[i] = arr1[linearPos];
	}
}

template<class T>
__global__ void GpuGeneralSArray(T *res,int resLength,int resBach,T *arr,int ofst1){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		T sum = 0;
		int ofst = i * ofst1;
		for (int j=0; j<ofst1; j++){
			sum = sum + arr[ofst + i];
		}
		res[i] = sum;
	}
	
}

template<class T>
__global__ void GpuGeneralPower(T *res,int resLength,int resBach,int resColumns,
	T *src,int srcRows){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int r = i/resColumns;
		int c = i%resColumns;
		int pw = r / srcRows + 1;
		res[i] = pow(src[(r%srcRows)*resColumns + c],pw);
	}
	
}

template<class T>
__global__ void GpuGeneralPw(T *res,int resLength,int resBach,double pw,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		res[i] = pow(src[i],pw);
	}
}

template<class T>
__global__ void GpuGeneralCos(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		res[i] = cos(src[i]);
	}
}

template<class T>
__global__ void GpuGeneralMul3D(T *res,int resLength,int resBach,T *src1,int src1Rows,
	int src1Columns,T *src2,int src2Columns){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int o1 = src1Rows * src2Columns;
	int o2 = src1Rows * src1Columns;
	int o3 = src1Columns * src2Columns;
	
	for (int i=from; i<to; i++){
		int d = i / o1;
		int tmp0 = i % o1;
		int r = tmp0 / src2Columns;
		int c = tmp0 % src2Columns;
		T sum = 0;

		for (int k=0; k<src1Columns; k++){
			sum = sum + src1[d*o2 + r*src1Columns + k]*src2[d*o3 + k*src2Columns + c];
		}
		res[i] = sum;
	}
}

//Rotation about x-axis
template<class T>
__global__ void GpuGeneralRx(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 9;
		int tmp0 = i % 9;
		int r = tmp0 / 3;
		int c = tmp0 % 3;

		if (r == 0){
			if (c == 0){
				res[i] = 1.0;
			}
			else{
				res[i] = 0.0;
			}
		}
		else if (r == 1){
			if (c == 0){
				res[i] = 0.0;
			}
			else if(c == 1){
				res[i] = cos(src[d]);
			}
			else{
				res[i] = -sin(src[d]);
			}
		}
		else{
			if (c == 0){
				res[i] = 0;
			}
			else if (c == 1){
				res[i] = sin(src[d]);
			}
			else{
				res[i] = cos(src[d]);
			}
		}
	}
}

//Rotation about y-axis
template<class T>
__global__ void GpuGeneralRy(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 9;
		int tmp0 = i % 9;
		int r = tmp0 / 3;
		int c = tmp0 % 3;

		if (r == 0){
			if (c == 0){
				res[i] = cos(src[d]);
			}
			else if (c == 1){
				res[i] = 0.0;
			}
			else{
				res[i] = sin(src[d]);
			}
		}
		else if (r == 1){
			if (c == 0){
				res[i] = 0.0;
			}
			else if(c == 1){
				res[i] = 1.0;
			}
			else{
				res[i] = 0.0;
			}
		}
		else{
			if (c == 0){
				res[i] = -sin(src[d]);
			}
			else if (c == 1){
				res[i] = 0.0;
			}
			else{
				res[i] = cos(src[d]);
			}
		}
	}
}

//Rotation about z-axis
template<class T>
__global__ void GpuGeneralRz(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 9;
		int tmp0 = i % 9;
		int r = tmp0 / 3;
		int c = tmp0 % 3;

		if (r == 0){
			if (c == 0){
				res[i] = cos(src[d]);
			}
			else if (c == 1){
				res[i] = -sin(src[d]);
			}
			else{
				res[i] = 0.0;
			}
		}
		else if (r == 1){
			if (c == 0){
				res[i] = sin(src[d]);
			}
			else if(c == 1){
				res[i] = cos(src[d]);
			}
			else{
				res[i] = 0.0;
			}
		}
		else{
			if (c == 0){
				res[i] = 0.0;
			}
			else if (c == 1){
				res[i] = 0.0;
			}
			else{
				res[i] = 1.0;
			}
		}
	}
}

// x
// 0
// 0
template<class T>
__global__ void GpuGeneralV3DX(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 3;
		int r = i % 3;

		if (r == 0){
			res[i] = src[d];
		}
		else if (r == 1){
			res[i] = 0.0;
		}
		else{
			res[i] = 0.0;
		}
	}
}

template<class T>
__global__ void GpuGeneralV3DY(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 3;
		int r = i % 3;

		if (r == 0){
			res[i] = 0.0;
		}
		else if (r == 1){
			res[i] = src[d];
		}
		else{
			res[i] = 0.0;
		}
	}
}

template<class T>
__global__ void GpuGeneralV3DZ(T *res,int resLength,int resBach,T *src){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int d = i / 3;
		int r = i % 3;

		if (r == 0){
			res[i] = 0.0;
		}
		else if (r == 1){
			res[i] = 0.0;
		}
		else{
			res[i] = src[d];
		}
	}
}

//d*r*1 -> r*d
template<class T>
__global__ void GpuGeneralFrom3DTo2D(T *res,int resLength,int resBach,T *src,int srcDepth,
	int srcRows){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	for (int i=from; i<to; i++){
		int r = i / srcDepth;
		int c = i % srcDepth;

		res[i] = src[c * srcRows + r];
	}
}

//d*n*m -> m*n*d
template<class T>
__global__ void GpuGeneralTr3DZ(T *res,int resLength,int resBach,T *src,int srcDepth,int srcRows,int srcColumns){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	int o = srcRows * srcDepth;

	for (int i=from; i<to; i++){
		int d = i / o;
		int tmp = i % o;
		int r = tmp / srcDepth;
		int c = tmp % srcDepth;

		res[i] = src[c * srcRows * srcColumns + r * srcColumns + d];
	}
}

template<class T>
__global__ void GpuGeneralTpu(T *res,int resLength,int resBach,T *jacobian,
	int jacobianDepth,int jacobianRows,int jacobianColumns,T *variance){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}
	
	int o = jacobianRows * jacobianColumns;

	for (int i=from; i<to; i++){
		int r = i / jacobianDepth;
		int c = i % jacobianDepth;

		T sum = 0;
		
		for (int k=0; k<jacobianColumns; k++){
			sum = sum + pow(jacobian[o*c + r * jacobianColumns + k]*variance[c * jacobianColumns + k],2);
		}
		res[i] = sqrt(sum);
	}
}

//
//St stack category
//
template<class T>
__global__ void GpuGeneral2DSt0(T *res,int resLength,int resBach,T *src1,int src1Columns,
	T *src2,int src2Columns){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	int columns = src1Columns + src2Columns;

	for (int i=from; i<to; i++){
		int r = i / columns;
		int c = i % columns;

		if (c < src1Columns){
			res[i] = src1[r*src1Columns + c];
		}
		else{
			res[i] = src2[r*src2Columns + c - src1Columns];
		}
	}
}

template<class T>
__global__ void GpuGeneral2DSt1(T *res,int resLength,int resBach,int resColumns,T *src1,
	int src1Rows,T *src2,int src2Rows){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		int r = i / resColumns;
		int c = i % resColumns;
		if (r < src1Rows){
			res[i] = src1[r*resColumns + c];
		}
		else{
			res[i] = src2[(r-src1Rows)*resColumns + c];
		}
	}
}

template<class T>
__global__ void GpuGeneral2DFourierSin(T *res,int resLength,int resBach,int resColumns,
	T *src,int srcRows){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		int r = i / resColumns;
		int c = i % resColumns;
		int freq = r / srcRows + 1;

		res[i] = sin(freq*src[(r%srcRows)*resColumns+c]);
	}
}

template<class T>
__global__ void GpuGeneral2DFourierCos(T *res,int resLength,int resBach,int resColumns,
	T *src,int srcRows){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		int r = i / resColumns;
		int c = i % resColumns;
		int freq = r / srcRows + 1;

		res[i] = cos(freq*src[(r%srcRows)*resColumns+c]);
	}
}

template<class T>
__global__ void GpuGeneral3DAlpha(T *res,int resLength,int resBach,T *src,int srcColumns){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	int frame = 3 * srcColumns;

	for (int i=from; i<to; i++){
		int r = i / 9;
		int c = i % 9;

		int ofst = r*frame;
		T sum = 0;

		if (c == 0){
			int ofstX = ofst;
			for (int i=0; i<srcColumns; i++){
				sum = sum + pow(src[ofstX + i],2);
			}
		}
		else if (c == 1){
			int ofstY = ofst + srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + pow(src[ofstY + i],2);
			}
		}
		else if (c == 2){
			int ofstZ = ofst + 2*srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + pow(src[ofstZ + i],2);
			}
		}
		else if (c == 3){
			int ofstX = ofst;
			int ofstY = ofst + srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstX + i] * src[ofstY + i];
			}
		}
		else if (c == 4){
			int ofstX = ofst;
			int ofstZ = ofst + 2*srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstX + i] * src[ofstZ + i];
			}
		}
		else if (c == 5){
			int ofstY = ofst + srcColumns;
			int ofstZ = ofst + 2*srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstY + i] * src[ofstZ + i];
			}
		}
		else if (c == 6){
			int ofstX = ofst;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstX + i];
			}
		}
		else if (c == 7){
			int ofstY = ofst + srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstY + i];
			}
		}
		else if (c == 8){
			int ofstZ = ofst + 2*srcColumns;
			for (int i=0; i<srcColumns; i++){
				sum = sum + src[ofstZ + i];
			}
		}
		res[i] = sum;
	}
}

template <class T>
__global__ void GpuGeneral3DPlaneL2(T *res,int resLength,int resBach,T *src,int samples){
	int from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;
	int to = from + resBach;

	if (to > resLength){
		to = resLength;
	}

	for (int i=from; i<to; i++){
		int ofstSrc = i * 9;
		int ofstRes = i * 3;
		T A0 = src[ofstSrc];
		T A1 = src[ofstSrc + 1];
		T A3 = src[ofstSrc + 3];
		T A4 = src[ofstSrc + 4];
		T A5 = src[ofstSrc + 5];
		T A6 = src[ofstSrc + 6];
		T A7 = src[ofstSrc + 7];
		T A8 = src[ofstSrc + 8];

		T tmp0 = A0*A1 - A3*A3;

		T gamma = (A8 - ((A4 * A6)/A0) + (A0*A5*A3*A6 - A0*A0*A5*A7 - A3*A3*A6*A4 + A4*A3*A7*A0)/tmp0)/(samples + 1 - (A6*A6)/A0 - (A3*A3*A6*A6 - A7*A7*A0*A0)/tmp0);
		T bita = (A0*A5 + gamma * A3 * A6 - A3*A4 + gamma*A7*A0)/tmp0;
		T alpha = (A4 - bita*A3 - gamma*A6)/A0;

		res[ofstRes] = alpha;
		res[ofstRes + 1] = bita;
		res[ofstRes + 2] = gamma;
	}
}

//GpuGeneral Functions

template<class T>
class LListNode{
	T *data;
	bool keep;		//if keep == false then when the object is deleted then the is deleted too, else the data is kept
	LListNode *next;
	public:
		LListNode(T *tData,bool tKeep = false){
			data = tData;
			keep = tKeep;
			next = NULL;
		}
		~LListNode(){
			if (!keep){
				delete data;
			}
		}
		T *GetData(){
			return data;
		}
		LListNode *GetNext(){
			return next;
		}
		void SetNext(LListNode *tNext){
			next = tNext;
		}
};

template<class T>
class LList{
	LListNode<T> *head;
	LListNode<T> *tail;
	bool rMode; 		//if rMode = true then stuff will be added to
				//the biging of the list
	public:
		LList(){
			head = NULL;
			tail = NULL;
			rMode = false;
		}
		~LList(){
			if (!this->IsEmpty()){
				LListNode<T> *temp1 = head;
				LListNode<T> *temp2 = temp1->GetNext();
				while(temp2 != NULL){
					delete temp1;
					temp1 = temp2;
					temp2 = temp2->GetNext();
				}
				delete temp1;
			}
		}
		bool IsEmpty(){
			return (head == NULL);
		}
		void Add(T *data,bool keep = false){
			LListNode<T> *nNode = new LListNode<T>(data,keep);
			if (head == NULL){
				head = nNode;
				tail = nNode;
			}
			else{
				if (!rMode){
					tail->SetNext(nNode);
					tail = nNode;
				}
				else{
					nNode->SetNext(head);
					head = nNode;
				}
			}
		}
		LListNode<T> *GetHead(){
			return head;
		}
		void SetRMode(){
			rMode = true;
		}
		int GetNumOfNodes(){
			int res = 0;
			LListNode<T> *temp = head;
			while(temp != NULL){
				res++;
				temp = temp->GetNext();
			}
			return res;
		}
};

template<class T>
class CpuVector{
	int length;
	int size;	//bytes
	T *p;
	public:
		CpuVector(){
			length = -1;
			size = -1;
			p = NULL;
		}
		CpuVector(int tLength){
			length = tLength;
			size = length * sizeof(T);
			p = (T *)malloc(size);
		}
		~CpuVector(){
			free(p);
		}
		void Allocate(int tLength){
			length = tLength;
			size = length * sizeof(T);
			p = (T *)malloc(size);
		}
		int GetLength(){
			return length;
		}
		int GetSize(){
			return size;
		}
		T *GetP(){
			return p;
		}
		void Set(int pos,T val){
			p[pos] = val;
		}
		T Get(int pos){
			return p[pos];
		}
		T GetLast(){
			return this->Get(length - 1);
		}
		T GetFirst(){
			return this->Get(0);
		}
		void SetLast(T val){
			this->Set(length - 1,val);
		}
		void SetFirst(T val){
			this->Set(0,val);
		}
		void Init(T val){
			for(int i=0; i<length; i++){
				p[i] = val;
			}
		}
};

template<class T>
class CpuArray{
	CpuVector<int> *dim;
	CpuVector<int> *ofst;
	int dimSize;
	int bytes;
	T *p;
	public:
		CpuArray(){
			dim = NULL;
			ofst = NULL;
			dimSize = -1;
			bytes = -1;
			p = NULL;
		}
		void Allocate(CpuVector<int> *tDim){
			dim = tDim;
			dimSize = dim->GetLength();
			ofst = new CpuVector<int>(dimSize + 1);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize - 1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			p = (T *)malloc(bytes);
		}
		CpuArray(CpuVector<int> *tDim){
			dim = tDim;
			dimSize = dim->GetLength();
			ofst = new CpuVector<int>(dimSize + 1);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize - 1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			p = (T *)malloc(bytes);
		}
		CpuArray(int tDimSize,...){
			va_list args;
			va_start(args,tDimSize);
			dimSize = tDimSize;
			dim = new CpuVector<int>(dimSize);
			ofst = new CpuVector<int>(dimSize + 1);
			for (int i=0; i<dimSize; i++){
				dim->Set(i,va_arg(args,int));
			}
			va_end(args);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize -1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			p = (T *)malloc(bytes);
		}
		~CpuArray(){
			delete dim;
			delete ofst;
			free(p);
		}
		CpuVector<int> *GetDim(){
			return dim;
		}
		CpuVector<int> *GetOfst(){
			return ofst;
		}
		int GetDimSize(){
			return dimSize;
		}
		int GetBytes(){
			return bytes;
		}
		T *GetP(){
			return p;
		}
		int GetLength(){
			return ofst->Get(0);
		}
		int GetPos(CpuVector<int> *pos){
			int res = 0;

			for (int i=0; i<dimSize; i++){
				res = res + pos->Get(i) * ofst->Get(i+1);
			}
			return res;
		}
		T Get(int d1,...){
			va_list args;
			va_start(args,d1);
			int linearPos = d1 * ofst->Get(1);
			for (int i=1; i<dimSize; i++){
				linearPos = linearPos + va_arg(args,int)*ofst->Get(i+1);
			}
			va_end(args);

			return p[linearPos];
		}
		T Get(CpuVector<int> *pos){
			int linearPos = this->GetPos(pos);

			return p[linearPos];
		}
		void Set(CpuVector<int> *pos,T val){
			int linearPos = this->GetPos(pos);
			p[linearPos] = val;
		}
		void Set(int d1,...){
			va_list args;
			va_start(args,d1);
			int linearPos = d1 * ofst->Get(1);
			for (int i=1; i<dimSize; i++){
				linearPos = linearPos + va_arg(args,int)*ofst->Get(i+1);
			}
			T val = va_arg(args,T);
			va_end(args);
			p[linearPos] = val;
		}
		void PrntSpaces(int k){
			for (int i=0; i<k; i++){
				std::cout<<" ";
			}
		}
		void  PrntH(int k,CpuVector<int> *pos){
			if (k == dimSize -1){
				this->PrntSpaces(k+1);
				std::cout<<"[";
				for (int i=0; i<dim->Get(k); i++){
					pos->Set(k,i);
					std::cout<<this->Get(pos)<<" ";
				}
				std::cout<<"]"<<"\n";
			}
			else{
				this->PrntSpaces(k+1);
				std::cout<<"["<<"\n";
				for (int i=0; i<dim->Get(k); i++){
					pos->Set(k,i);
					this->PrntH(k+1,pos);
				}
				std::cout<<"\n";
				this->PrntSpaces(k+1);
				std::cout<<"]"<<"\n";
			}
		}
		void Prnt(){
			CpuVector<int> *pos = new CpuVector<int>(dimSize);
			std::cout<<"["<<"\n";
			this->PrntH(0,pos);
			std::cout<<"\n"<<"]"<<"\n";
			delete pos;
		}
		void Init(T val){
			for (int i=0; i<ofst->Get(0); i++){
				p[i] = val;
			}
		}
};

//GVector
template<class T>
class GVector{
	int size;
	T **p;
	public:
		GVector(){
			size = -1;
			p = NULL;
		}
		GVector(int tSize){
			size = tSize;
			p = (T **)malloc(size * sizeof(T *));
		}
		~GVector(){
			for (int i=0; i<size; i++){
				delete p[i];
			}
			free(p);
		}
		void Allocate(int tSize){
			size = tSize;
			p = (T **)malloc(size * sizeof(T *));
		}
		T* Get(int pos) {
			return p[pos];
		}
		T* operator()(int pos){
			return p[pos];
		}
		void Set(int pos,T* val){
			p[pos] = val;
		}
		void operator()(int pos,T* val){
			p[pos] = val;
		}
		int GetSize(){return size;}
};
//GVector

template<class T>
class GpuVector{
	int length;
	int size;	//bytes
	T *p;
	public:
		GpuVector(){
			length = -1;
			size = -1;
			p = NULL;
		}
		GpuVector(int tLength){
			length = tLength;
			size = length * sizeof(T);
			cudaMalloc((void **)&p,size);
		}
		~GpuVector(){
			cudaFree(p);
		}
		void Allocate(int tLength){
			length = tLength;
			size = length * sizeof(T);
			cudaMalloc((void **)&p,size);
		}
		int GetLength(){
			return length;
		}
		int GetSize(){
			return size;
		}
		T *GetP(){
			return p;
		}
		void Set(int pos,T val){
			T *cpuP = (T *)malloc(sizeof(T));
			*cpuP = val;
			cudaMemcpy(p+pos,cpuP,sizeof(T),cudaMemcpyHostToDevice);
			free(cpuP);
		}
		T Get(int pos){
			T *cpuP = (T *)malloc(sizeof(T));
			cudaMemcpy(cpuP,p + pos,sizeof(T),cudaMemcpyDeviceToHost);
			T res = *cpuP;
			free(cpuP);

			return res;
		}
		T GetLast(){
			return this->Get(length - 1);
		}
		T GetFirst(){
			return this->Get(0);
		}
		void SetLast(T val){
			this->Set(length - 1,val);
		}
		void SetFirst(T val){
			this->Set(0,val);
		}
};

template<class T>
class GpuVectorOp{
	int blocks;
	int threads;
	public:
		GpuVectorOp(int tBlocks,int tThreads){
			blocks = tBlocks;
			threads = tThreads;
		}
		~GpuVectorOp(){}
		void Init(GpuVector<T> *v,T val){
			int thrds = blocks * threads;
			int length = v->GetLength();
			int bach = (length + thrds - 1)/thrds;
			GpuGeneralInit<T><<<blocks,threads>>>(v->GetP(),length,bach,val);
		}
		bool Equal(GpuVector<T> *v1,GpuVector<T> *v2){
			bool res = false;
			int s1 = v1->GetLength();
			int s2 = v2->GetLength();
			
			if (s1 == s2){
				int *gpuRes,*cpuOne;

				cpuOne = (int *)malloc(sizeof(int));
				*cpuOne = 1;
				cudaMalloc((void **)&gpuRes,sizeof(int));
				cudaMemcpy(gpuRes,cpuOne,sizeof(int),cudaMemcpyHostToDevice);
				int bach = (s1 + threads - 1)/threads;
				GpuGeneralEqual<T><<<1,threads>>>(gpuRes,v1->GetP(),s1,bach,v2->GetP());
				cudaMemcpy(cpuOne,gpuRes,sizeof(int),cudaMemcpyDeviceToHost);
				if (*cpuOne == 1){
					res = true;
				}
				free(cpuOne);
				cudaFree(gpuRes);
			}

			return res;
		}
		GpuVector<T> *Cpy(GpuVector<T> *v){
			int length = v->GetLength();
			GpuVector<T> *res = new GpuVector<T>(length);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			GpuGeneralCpy<T><<<blocks,threads>>>(res->GetP(),length,bach,v->GetP());

			return res;
		}
		GpuVector<T> *Reverse(GpuVector<T> *v){
			int length = v->GetLength();
			GpuVector<T> *res = new GpuVector<T>(length);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			GpuGeneralReverse<T><<<blocks,threads>>>(res->GetP(),length,bach,v->GetP());

			return res;
		}
		GpuVector<T> *Cpy(CpuVector<T> *v){
			GpuVector<T> *res = new GpuVector<T>(v->GetLength());
			cudaMemcpy(res->GetP(),v->GetP(),v->GetSize(),cudaMemcpyHostToDevice);

			return res;
		}
};


template<class T>
class GpuArray{
	GpuVector<int> *dim;
	GpuVector<int> *ofst;
	int dimSize;
	int bytes;
	T *p;
	public:
		GpuArray(){
			dim = NULL;
			ofst = NULL;
			dimSize = -1;
			bytes = -1;
			p = NULL;
		}
		void Allocate(GpuVector<int> *tDim){
			dim = tDim;
			dimSize = dim->GetLength();
			ofst = new GpuVector<int>(dimSize + 1);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize - 1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			cudaMalloc((void **)&p,bytes);
		}
		GpuArray(GpuVector<int> *tDim){
			dim = tDim;
			dimSize = dim->GetLength();
			ofst = new GpuVector<int>(dimSize + 1);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize - 1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			cudaMalloc((void **)&p,bytes);
		}
		GpuArray(int tDimSize,...){
			va_list args;
			va_start(args,tDimSize);
			dimSize = tDimSize;
			dim = new GpuVector<int>(dimSize);
			ofst = new GpuVector<int>(dimSize + 1);
			for (int i=0; i<dimSize; i++){
				dim->Set(i,va_arg(args,int));
			}
			va_end(args);
			int accum = 1;
			ofst->Set(dimSize,accum);
			for (int i=dimSize -1; i>=0; i--){
				accum = accum * dim->Get(i);
				ofst->Set(i,accum);
			}
			bytes = accum * sizeof(T);
			cudaMalloc((void **)&p,bytes);
		}
		~GpuArray(){
			delete dim;
			delete ofst;
			cudaFree(p);
		}
		GpuVector<int> *GetDim(){
			return dim;
		}
		GpuVector<int> *GetOfst(){
			return ofst;
		}
		int GetDimSize(){
			return dimSize;
		}
		int GetBytes(){
			return bytes;
		}
		T *GetP(){
			return p;
		}
		int GetLength(){
			return ofst->Get(0);
		}
		int GetPos(GpuVector<int> *pos){
			int res = 0;

			for (int i=0; i<dimSize; i++){
				res = res + pos->Get(i) * ofst->Get(i+1);
			}
			return res;
		}
		T Get(int d1,...){
			va_list args;
			va_start(args,d1);
			int linearPos = d1 * ofst->Get(1);
			for (int i=1; i<dimSize; i++){
				linearPos = linearPos + va_arg(args,int)*ofst->Get(i+1);
			}
			va_end(args);
			T *cpuP = (T *)malloc(sizeof(T));
			cudaMemcpy(cpuP,p + linearPos,sizeof(T),cudaMemcpyDeviceToHost);
			T res = *cpuP;
			free(cpuP);

			return res;
		}
		T Get(GpuVector<int> *pos){
			int linearPos = this->GetPos(pos);
			T *cpuP = (T *)malloc(sizeof(T));
			cudaMemcpy(cpuP,p + linearPos,sizeof(T),cudaMemcpyDeviceToHost);
			T res = *cpuP;
			free(cpuP);

			return res;
		}
		void Set(GpuVector<int> *pos,T val){
			int linearPos = this->GetPos(pos);
			T *cpuP = (T *)malloc(sizeof(T));
			*cpuP = val;
			cudaMemcpy(p + linearPos,cpuP,sizeof(T),cudaMemcpyHostToDevice);
			free(cpuP);
		}
		void Set(int d1,...){
			va_list args;
			va_start(args,d1);
			int linearPos = d1 * ofst->Get(1);
			for (int i=1; i<dimSize; i++){
				linearPos = linearPos + va_arg(args,int)*ofst->Get(i+1);
			}
			T val = va_arg(args,T);
			va_end(args);
			T *cpuP = (T *)malloc(sizeof(T));
			*cpuP = val;
			cudaMemcpy(p + linearPos,cpuP,sizeof(T),cudaMemcpyHostToDevice);
			free(cpuP);
		}
		void PrntSpaces(int k){
			for (int i=0; i<k; i++){
				std::cout<<" ";
			}
		}
		void  PrntH(int k,GpuVector<int> *pos){
			if (k == dimSize -1){
				this->PrntSpaces(k+1);
				std::cout<<"[";
				for (int i=0; i<dim->Get(k); i++){
					pos->Set(k,i);
					std::cout<<this->Get(pos)<<" ";
				}
				std::cout<<"]"<<"\n";
			}
			else{
				this->PrntSpaces(k+1);
				std::cout<<"["<<"\n";
				for (int i=0; i<dim->Get(k); i++){
					pos->Set(k,i);
					this->PrntH(k+1,pos);
				}
				std::cout<<"\n";
				this->PrntSpaces(k+1);
				std::cout<<"]"<<"\n";
			}
		}
		void Prnt(){
			GpuVector<int> *pos = new GpuVector<int>(dimSize);
			std::cout<<"["<<"\n";
			this->PrntH(0,pos);
			std::cout<<"\n"<<"]"<<"\n";
			delete pos;
		}
		void Add(GpuArray<T> *arr,int blocks = 1024,int threads = 1024){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(dim,arrDim)){
				GpuGeneralAdd<T><<<blocks,threads>>>(p,length,bach,p,
					arr->GetP());
			}
			else{
				GpuVector<int> *arrOfst = arr->GetOfst();

				GpuGeneralAddBcast<T><<<blocks,threads>>>(p,length,bach,p,
					dim->GetP(),dim->GetLength(),ofst->GetP(),arr->GetP(),
					arrDim->GetP(),arrDim->GetLength(),arrOfst->GetP());
			}
		}
		void Add(T val,int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralAddCnst<T><<<blocks,threads>>>(p,length,bach,p,val);
		}
		void Sub(GpuArray<T> *arr,int blocks = 1024,int threads = 1024){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(dim,arrDim)){
				GpuGeneralSub<T><<<blocks,threads>>>(p,length,bach,p,
					arr->GetP());
			}
			else{
				GpuVector<int> *arrOfst = arr->GetOfst();

				GpuGeneralSubBcast<T><<<blocks,threads>>>(p,length,bach,p,
					dim->GetP(),dim->GetLength(),ofst->GetP(),arr->GetP(),
					arrDim->GetP(),arrDim->GetLength(),arrOfst->GetP());
			}
		}
		void Sub(T val,int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralSubCnst<T><<<blocks,threads>>>(p,length,bach,p,val);
		}
		void Dot(GpuArray<T> *arr,int blocks = 1024,int threads = 1024){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(dim,arrDim)){
				GpuGeneralDot<T><<<blocks,threads>>>(p,length,bach,p,
					arr->GetP());
			}
			else{
				GpuVector<int> *arrOfst = arr->GetOfst();

				GpuGeneralDotBcast<T><<<blocks,threads>>>(p,length,bach,p,
					dim->GetP(),dim->GetLength(),ofst->GetP(),arr->GetP(),
					arrDim->GetP(),arrDim->GetLength(),arrOfst->GetP());
			}
		}
		void Dot(T val,int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralDotCnst<T><<<blocks,threads>>>(p,length,bach,p,val);
		}
		void Div(GpuArray<T> *arr,int blocks = 1024,int threads = 1024){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(dim,arrDim)){
				GpuGeneralDiv<T><<<blocks,threads>>>(p,length,bach,p,
					arr->GetP());
			}
			else{
				GpuVector<int> *arrOfst = arr->GetOfst();

				GpuGeneralDivBcast<T><<<blocks,threads>>>(p,length,bach,p,
					dim->GetP(),dim->GetLength(),ofst->GetP(),arr->GetP(),
					arrDim->GetP(),arrDim->GetLength(),arrOfst->GetP());
			}
		}
		void Div(T val,int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralDivCnst<T><<<blocks,threads>>>(p,length,bach,p,val);
		}
		void Sigmoid(int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralSigmoid<T><<<blocks,threads>>>(p,length,bach,p);
		}
		void Log(int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralLog<T><<<blocks,threads>>>(p,length,bach,p);

		}
		void Pw(double pw,int blocks = 1024,int threads = 1024){
			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralPw<T><<<blocks,threads>>>(p,length,bach,pw,p);
		}
		T Sum(){
			T *gpuRes,*cpuRes;
			cudaMalloc((void **)&gpuRes,sizeof(T));
			cpuRes = (T *)malloc(sizeof(T));
			*cpuRes = 0;
			cudaMemcpy(gpuRes,cpuRes,sizeof(T),cudaMemcpyHostToDevice);
			GpuGeneralSum<T><<<1,1>>>(gpuRes,p,ofst->Get(0));
			cudaMemcpy(cpuRes,gpuRes,sizeof(T),cudaMemcpyDeviceToHost);
			T res = *cpuRes;
			free(cpuRes);
			cudaFree(cpuRes);

			return res;
		}
		//arr is a 2D array
		void Set3D(GpuArray<T> *arr,int pos,int blocks = 1024,int threads = 1024){
			int rows = dim->Get(1);
			int columns = dim->Get(2);
			
			int length = rows * columns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralCpy<T><<<blocks,threads>>>(p + length*pos,length,bach,arr->GetP());
		}

};


template<class T>
class GpuArrayOp{
	int blocks; 
	int threads;
	public:
		GpuArrayOp(int tBlocks,int tThreads){
			blocks = tBlocks;
			threads = tThreads;
		}
		~GpuArrayOp(){}
		void SetSrand(unsigned int seed){
			srand(seed);
		}
		void Init(GpuArray<T> *arr,T val){
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;
			GpuGeneralInit<T><<<blocks,threads>>>(arr->GetP(),length,bach,val);
		}
		void LinearInit(GpuArray<T> *arr){
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralLinearInit<T><<<blocks,threads>>>(arr->GetP(),length,bach);
		}
		GpuArray<T> *Rnd(int d,...){
			va_list args;

			va_start(args,d);
			GpuVector<int> *dim = new GpuVector<int>(d);

			for (int i=0; i<d; i++){
				dim->Set(i,va_arg(args,int));
			}

			T from = va_arg(args,T);
			T to = va_arg(args,T);

			va_end(args);

			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = res->GetLength();
			int bytes = res->GetBytes();
			T val;
			T *cpuRes = (T *)malloc(bytes);

			for (int i=0; i<length; i++){
				val = (T)(from + ((double)rand()/RAND_MAX)*(to - from));
				cpuRes[i] = val;
			}
			cudaMemcpy(res->GetP(),cpuRes,bytes,cudaMemcpyHostToDevice);

			free(cpuRes);

			return res;
		}
		GpuArray<T> *Mul(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			GpuVector<int> *arr1Ofst = arr1->GetOfst();

			int resRows = arr1Dim->Get(0);
			int resColumns = arr2Dim->GetLast();
			int resLength = resRows * resColumns;
			int thrds = blocks * threads;
			int bach = (resLength + thrds - 1)/thrds;
			int commonDim = arr1Ofst->Get(1);
			GpuVector<int> *resDim = new GpuVector<int>(2);
			resDim->Set(0,resRows);
			resDim->Set(1,resColumns);
			GpuArray<T> *res = new GpuArray<T>(resDim);

			GpuGeneralMul<T><<<blocks,threads>>>(res->GetP(),resLength,bach,resColumns,commonDim
				,arr1->GetP(),arr2->GetP());

			return res;
		}
		GpuArray<T> *Add(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);

			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(arr1Dim,arr2Dim)){
				GpuGeneralAdd<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr2->GetP());
			}
			else{
				GpuVector<int> *arr1Ofst = arr1->GetOfst();
				GpuVector<int> *arr2Ofst = arr2->GetOfst();

				GpuGeneralAddBcast<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr1Dim->GetP(),arr1Dim->GetLength(),arr1Ofst->GetP(),arr2->GetP(),
					arr2Dim->GetP(),arr2Dim->GetLength(),arr2Ofst->GetP());
			}

			return res;
		}
		GpuArray<T> *Add(GpuArray<T> *arr1,T val){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralAddCnst<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),val);

			return res;
		}
		GpuArray<T> *Sub(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);

			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(arr1Dim,arr2Dim)){
				GpuGeneralSub<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr2->GetP());
			}
			else{
				GpuVector<int> *arr1Ofst = arr1->GetOfst();
				GpuVector<int> *arr2Ofst = arr2->GetOfst();

				GpuGeneralSubBcast<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr1Dim->GetP(),arr1Dim->GetLength(),arr1Ofst->GetP(),arr2->GetP(),
					arr2Dim->GetP(),arr2Dim->GetLength(),arr2Ofst->GetP());
			}

			return res;
		}
		GpuArray<T> *Sub(GpuArray<T> *arr1,T val){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralSubCnst<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),val);

			return res;
		}
		GpuArray<T> *Div(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);

			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(arr1Dim,arr2Dim)){
				GpuGeneralDiv<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr2->GetP());
			}
			else{
				GpuVector<int> *arr1Ofst = arr1->GetOfst();
				GpuVector<int> *arr2Ofst = arr2->GetOfst();

				GpuGeneralDivBcast<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr1Dim->GetP(),arr1Dim->GetLength(),arr1Ofst->GetP(),arr2->GetP(),
					arr2Dim->GetP(),arr2Dim->GetLength(),arr2Ofst->GetP());
			}

			return res;
		}
		GpuArray<T> *Div(GpuArray<T> *arr1,T val){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralDivCnst<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),val);

			return res;
		}
		GpuArray<T> *Dot(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);

			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1 )/thrds;

			if (op.Equal(arr1Dim,arr2Dim)){
				GpuGeneralDot<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr2->GetP());
			}
			else{
				GpuVector<int> *arr1Ofst = arr1->GetOfst();
				GpuVector<int> *arr2Ofst = arr2->GetOfst();

				GpuGeneralDotBcast<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
					arr1Dim->GetP(),arr1Dim->GetLength(),arr1Ofst->GetP(),arr2->GetP(),
					arr2Dim->GetP(),arr2Dim->GetLength(),arr2Ofst->GetP());
			}

			return res;
		}
		GpuArray<T> *Dot(GpuArray<T> *arr1,T val){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *resDim = op.Cpy(arr1Dim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr1->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneralDotCnst<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),val);

			return res;
		}
		GpuArray<T> *Sigmoid(GpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *resDim = op.Cpy(arrDim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralSigmoid<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP());

			return res;
		}
		GpuArray<T> *Log(GpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *resDim = op.Cpy(arrDim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralLog<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP());

			return res;
		}
		GpuArray<T> *Transpose(GpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *resDim = op.Reverse(arrDim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			GpuVector<int> *arrOfst = arr->GetOfst();
			GpuVector<int> *resOfst = res->GetOfst();
		
			GpuGeneralTranspose<T><<<blocks,threads>>>(res->GetP(),length,bach,
				resOfst->GetP(),arr->GetP(),arrDim->GetLength(),arrOfst->GetP());

			return res;
		}
		GpuArray<T> *Cpy(GpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);

			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *resDim = op.Cpy(arrDim);
			GpuArray<T> *res = new GpuArray<T>(resDim);
			int length = arr->GetLength();
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralCpy<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP());

			return res;
		}
		GpuArray<T> *Cpy(CpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);
			CpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *dim = op.Cpy(arrDim);
			GpuArray<T> *res = new GpuArray<T>(dim);
			cudaMemcpy(res->GetP(),arr->GetP(),res->GetBytes(),cudaMemcpyHostToDevice);

			return res;
		}
		GpuArray<T> *SArray(GpuArray<T> *arr){
			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *arrOfst = arr->GetOfst();
			int length = arrDim->Get(0);
			int ofst1 = arrOfst->Get(1);
			GpuArray<T> *res = new GpuArray<T>(2,length,1);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			GpuGeneralSArray<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP(),ofst1);

			return res;
		}
		GpuArray<T> *Power(GpuArray<T> *arr,int power){
			GpuVector<int> *arrDim = arr->GetDim();
			int srcRows = arrDim->Get(0);
			int srcColumns = arrDim->Get(1);
			int resRows = srcRows * power;
			GpuArray<T> *res = new GpuArray<T>(2,resRows,srcColumns);
			int length = resRows * srcColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			GpuGeneralPower<T><<<blocks,threads>>>(res->GetP(),length,bach,srcColumns,
				arr->GetP(),srcRows);
			return res;
		}
		GpuArray<T> *Pw(GpuArray<T> *arr,double pw){
			GpuVectorOp<int> op(blocks,threads);
			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *dim = op.Cpy(arrDim);
			GpuVector<int> *ofst = arr->GetOfst();
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralPw<T><<<blocks,threads>>>(res->GetP(),length,bach,pw,arr->GetP());

			return res;
		}
		GpuArray<T> *Cos(GpuArray<T> *arr){
			GpuVectorOp<int> op(blocks,threads);
			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *dim = op.Cpy(arrDim);
			GpuVector<int> *ofst = arr->GetOfst();
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = ofst->Get(0);
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralCos<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP());

			return res;
		}
		GpuArray<T> *Mul3D(GpuArray<T> *arr1,GpuArray<T> *arr2){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			GpuVector<int> *arr2Dim = arr2->GetDim();
			int depth = arr1Dim->Get(0);
			int arr1Rows = arr1Dim->Get(1);
			int arr1Columns = arr1Dim->Get(2);
			int arr2Columns = arr2Dim->Get(2);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,depth);
			dim->Set(1,arr1Rows);
			dim->Set(2,arr2Columns);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = depth * arr1Rows * arr2Columns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralMul3D<T><<<blocks,threads>>>(res->GetP(),length,bach,arr1->GetP(),
				arr1Rows,arr1Columns,arr2->GetP(),arr2Columns);

			return res;
		}
		//arr1 is a 2d array, m is the row
		GpuArray<T> *Rx(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,3);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 9;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralRx<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		GpuArray<T> *Ry(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,3);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 9;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralRy<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		GpuArray<T> *Rz(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,3);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 9;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralRz<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		GpuArray<T> *V3DX(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,1);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 3;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralV3DX<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		GpuArray<T> *V3DY(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,1);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 3;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralV3DY<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		GpuArray<T> *V3DZ(GpuArray<T> *arr1,int m){
			GpuVector<int> *arr1Dim = arr1->GetDim();
			int arr1Columns = arr1Dim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(3);
			dim->Set(0,arr1Columns);
			dim->Set(1,3);
			dim->Set(2,1);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Columns * 3;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralV3DZ<T><<<blocks,threads>>>(res->GetP(),length,bach,
				arr1->GetP() + m * arr1Columns);

			return res;
		}
		//Assumes that arr is n*m*1 .Result m*n
		GpuArray<T> *From3DTo2D(GpuArray<T> *arr){
			GpuVector<int> *arrDim = arr->GetDim();
			int arr1Depth = arrDim->Get(0);
			int arr1Rows = arrDim->Get(1);
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,arr1Rows);
			dim->Set(1,arr1Depth);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arr1Depth * arr1Rows;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralFrom3DTo2D<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP(),
				arr1Depth,arr1Rows);

			return res;
		}
		//Assumes 2D array,adds to r row dx.
		GpuArray<T> *AddToRow(GpuArray<T> *arr,int r,T dx){
			GpuVector<int> *arrDim = arr->GetDim();
			int arrColumns = arrDim->Get(1);
			GpuArray<T> *res = this->Cpy(arr);

			int length = arrColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralAddCnst<T><<<blocks,threads>>>(res->GetP() + r*arrColumns,length,bach,res->GetP() + r*arrColumns,dx);

			return res;
		}
		GpuArray<T> *Tr3DZ(GpuArray<T> *arr){
			GpuVector<int> *arrDim = arr->GetDim();
			GpuVector<int> *dim = new GpuVector<int>(3);
			int arrDepth = arrDim->Get(0);
			int arrRows = arrDim->Get(1);
			int arrColumns = arrDim->Get(2);
			dim->Set(0,arrColumns);
			dim->Set(1,arrRows);
			dim->Set(2,arrDepth);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = arrDepth * arrRows * arrColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralTr3DZ<T><<<blocks,threads>>>(res->GetP(),length,bach,arr->GetP(),
				arrDepth,arrRows,arrColumns);

			return res;
		}
		//jacobian 3D array, variance 2D array
		//d * n * m , d * m
		GpuArray<T> *Tpu(GpuArray<T> *jacobian,GpuArray<T> *variance){
			GpuVector<int> *jacobianDim = jacobian->GetDim();
			int jacobianDepth = jacobianDim->Get(0);
			int jacobianRows = jacobianDim->Get(1);
			int jacobianColumns = jacobianDim->Get(2);
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,jacobianRows);
			dim->Set(1,jacobianDepth);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = jacobianRows * jacobianDepth;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneralTpu<T><<<blocks,threads>>>(res->GetP(),length,bach,jacobian->GetP(),jacobianDepth,jacobianRows,jacobianColumns,variance->GetP());

			return res;
		}
		GpuArray<T> *_2DSt0(GpuArray<T> *src1,GpuArray<T> *src2){
			GpuVector<int> *src1Dim = src1->GetDim();
			GpuVector<int> *src2Dim = src2->GetDim();
			int rows = src1Dim->Get(0);
			int src1Columns = src1Dim->Get(1);
			int src2Columns = src2Dim->Get(1);
			int resColumns = src1Columns + src2Columns;
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,rows);
			dim->Set(1,resColumns);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = rows * resColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds -1)/thrds;

			GpuGeneral2DSt0<T><<<blocks,threads>>>(res->GetP(),length,bach,src1->GetP(),src1Columns,src2->GetP(),
				src2Columns);

			return res;
		}
		GpuArray<T> *_2DSt1(GpuArray<T> *src1,GpuArray<T> *src2){
			GpuVector<int> *src1Dim = src1->GetDim();
			GpuVector<int> *src2Dim = src2->GetDim();
			int src1Rows = src1Dim->Get(0);
			int src2Rows = src2Dim->Get(0);
			int resColumns = src1Dim->Get(1);
			int resRows = src1Rows + src2Rows;
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,resRows);
			dim->Set(1,resColumns);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = resRows * resColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneral2DSt1<T><<<blocks,threads>>>(res->GetP(),length,bach,resColumns,
				src1->GetP(),src1Rows,src2->GetP(),src2Rows);

			return res;
		}
		GpuArray<T> * _2DFourierSin(GpuArray<T> *src,int freq){
			GpuVector<int> *srcDim = src->GetDim();
			int srcRows = srcDim->Get(0);
			int srcColumns = srcDim->Get(1);
			int resRows = srcRows * freq;
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,resRows);
			dim->Set(1,srcColumns);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = resRows * srcColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneral2DFourierSin<T><<<blocks,threads>>>(res->GetP(),length,bach,srcColumns,
				src->GetP(),srcRows);

			return res;
		}
		GpuArray<T> * _2DFourierCos(GpuArray<T> *src,int freq){
			GpuVector<int> *srcDim = src->GetDim();
			int srcRows = srcDim->Get(0);
			int srcColumns = srcDim->Get(1);
			int resRows = srcRows * freq;
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,resRows);
			dim->Set(1,srcColumns);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = resRows * srcColumns;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneral2DFourierCos<T><<<blocks,threads>>>(res->GetP(),length,bach,srcColumns,
				src->GetP(),srcRows);

			return res;
		}
		GpuArray<T> * _3DAlpha(GpuArray<T> *src){
			GpuVector<int> *srcDim = src->GetDim();
			int srcDepth = srcDim->Get(0);
			int srcColumns = srcDim->Get(2);
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,srcDepth);
			dim->Set(1,9);
			GpuArray<T> *res = new GpuArray<T>(dim);

			int length = srcDepth * 9;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;
			
			GpuGeneral3DAlpha<T><<<blocks,threads>>>(res->GetP(),length,bach,src->GetP(),
				srcColumns);

			return res;
		}
		GpuArray<T> * _3DPlaneL2(GpuArray<T> *src){
			GpuVector<int> *srcDim = src->GetDim();
			int samples = srcDim->Get(2);
			GpuArray<T> *alpha = this->_3DAlpha(src);
			GpuVector<int> *alphaDim = alpha->GetDim();
			int alphaRows = alphaDim->Get(0);
			GpuVector<int> *dim = new GpuVector<int>(2);
			dim->Set(0,alphaRows);
			dim->Set(1,3);
			GpuArray<T> *res = new GpuArray<T>(dim);
			
			int length = alphaRows;
			int thrds = blocks * threads;
			int bach = (length + thrds - 1)/thrds;

			GpuGeneral3DPlaneL2<T><<<blocks,threads>>>(res->GetP(),length,bach,src->GetP(),samples);

			delete alpha;
			return res;
		}
};


class DNNSigmoid{
	GVector <GpuArray <double> > weights;
	GVector <GpuArray <double> > offsets;
	GpuVector<int> arch;
	int layers;
	int blocks;
	int threads;
	double learningRate;

	public:	
		DNNSigmoid(int tLayers,...){
			layers = tLayers;
			weights.Allocate(layers -1);
			offsets.Allocate(layers -1);
			arch.Allocate(layers);
			learningRate = 0.001;

			va_list args;
			va_start(args,tLayers);

			for (int i=0; i<layers; i++){
				arch.Set(i,va_arg(args,int));
			}

			blocks = va_arg(args,int);
			threads = va_arg(args,int);
			va_end(args);

			GpuArrayOp<double> op(blocks,threads);
			op.SetSrand(time(NULL));
			GpuArray<double> *w,*b;

			for (int i=0; i<layers - 1; i++){
				w = op.Rnd(2,arch.Get(i+1),arch.Get(i),-0.001,0.001);
				b = op.Rnd(2,arch.Get(i+1),1,-1.0,1.0);
				weights.Set(i,w);
				offsets.Set(i,b);
			}
		}
		~DNNSigmoid(){
		}
		void PrntWeights(){
			for (int i=0; i<layers -1; i++){
				weights.Get(i)->Prnt();
			}
		}
		void PrntOffsets(){
			for (int i=0; i<layers -1; i++){
				offsets.Get(i)->Prnt();
			}
		}
		GpuArray<double> *FProp(GpuArray<double> *input){
			GpuArray<double> *w,*b,*temp;
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *res = op.Cpy(input);
			
			for (int i=0; i<layers - 1; i++){
				w = weights.Get(i);
				b = offsets.Get(i);
				temp = res;
				res = op.Mul(w,res);
				res->Add(b);
				res->Sigmoid();
				delete temp;
			}
			return res;
		}
		double Loss(GpuArray<double> *input,GpuArray<double> *yHat){
			GpuArray<double> *y = this->FProp(input);
			GpuArray<double> *t1,*t2;
			GpuArrayOp<double> op(blocks,threads);
			t1 = op.Log(y); //t1 = log(y)
			y->Dot(-1.0);	
			y->Add(1.0);
			y->Log(blocks,threads); //y = log(1 - y)
			t2 = op.Dot(yHat,-1.0);
			t2->Add(1.0);			//t2 = 1 - yHat
			t1->Dot(yHat);
			t2->Dot(y);
			t2->Add(t1);

			GpuVector<int> *dim = yHat->GetDim();
			int columns = dim->Get(1);
			double res = (double)-(t2->Sum()/columns);

			delete y;
			delete t1;
			delete t2;

			return res;
		}
		double CorrectWeight(GpuArray<double> *input,GpuArray<double> *yHat,double pLoss,
			int x1,int x2,int x3){
			GpuArray<double> *w = weights.Get(x1);
			double val = w->Get(x2,x3);
			double uVal = val + learningRate;
			double dVal = val - learningRate;
			w->Set(x2,x3,uVal);
			double loss = this->Loss(input,yHat);
			if (pLoss < loss){
				w->Set(x2,x3,dVal);
				loss = this->Loss(input,yHat);
			}

			return loss;
		}
		double CorrectOffset(GpuArray<double> *input,GpuArray<double> *yHat,double pLoss,
			int x1,int x2){
			GpuArray<double> *b = offsets.Get(x1);
			double val = b->Get(x2,0);
			double uVal = val + learningRate;
			double dVal = val - learningRate;
			b->Set(x2,0,uVal);
			double loss = this->Loss(input,yHat);
			if (pLoss < loss){
				b->Set(x2,0,dVal);
				loss = this->Loss(input,yHat);
			}

			return loss;
		}
		double TrainOneBForse(GpuArray<double> *input,GpuArray<double> *yHat){
			double loss = this->Loss(input,yHat);
			GpuArray<double> *w;
			GpuVector<int> *dim;
			int rows,columns;
			
			for (int x1=0; x1<layers - 1; x1++){
				w = weights.Get(x1);
				dim = w->GetDim();
				rows = dim->Get(0);
				columns = dim->Get(1);
				for (int x2=0; x2<rows; x2++){
					for (int x3=0; x3<columns; x3++){
						loss = this->CorrectWeight(input,yHat,loss,x1,x2,x3);
					}
				}
			}
			
			GpuArray<double> *b;
			GpuVector<int> *bDim;

			for (int x1=0; x1<layers - 1; x1++){
				b = offsets.Get(x1);
				bDim = b->GetDim();
				rows = bDim->Get(0);
				for (int x2=0; x2<rows; x2++){
					loss = this->CorrectOffset(input,yHat,loss,x1,x2);
				}
			}

			return loss;
		}
		void TrainBForse(GpuArray<double> *input,GpuArray<double> *yHat,int epohs){
			for (int i=0; i<epohs; i++){
				double loss;
				loss = this->TrainOneBForse(input,yHat);
				std::cout<<loss<<"\n";
			}
		}
		GVector<GpuArray <double> > *FPropB(GpuArray<double> *input){
			GVector<GpuArray <double> > *res = new GVector< GpuArray <double> >(layers);
			GpuArray<double> *w,*b;
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *temp = op.Cpy(input);
			
			res->Set(0,temp);
			for (int i=1; i<layers; i++){
				w = weights.Get(i-1);
				b = offsets.Get(i-1);
				temp = op.Mul(w,temp);
				temp->Add(b);
				temp->Sigmoid();
				res->Set(i,temp);
			}

			return res;
		}
		GpuArray<double> *BPropH(GVector<GpuArray <double> > *pRes,GpuArray<double> *dz,int i){
			if (i >= 1){
				GpuArray<double> *fR,*pR,*d,*dzPrev,*w,*wT,*b,*db,*prT,*dw;
				GpuArrayOp<double> op(blocks,threads);
				fR = pRes->Get(i);
				pR = pRes->Get(i-1);
				w = weights.Get(i-1);
				b = offsets.Get(i-1);
				d = op.Dot(fR,-1.0);
				d->Add(1.0);
				d->Dot(fR);
				d->Dot(dz);
				delete dz;
				prT = op.Transpose(pR);
				dw = op.Mul(d,prT);
				dw->Dot(-learningRate);
				delete prT;
				w->Add(dw);
				delete dw;
				wT = op.Transpose(w);
				dzPrev = op.Mul(wT,d);
				delete wT;
				db = op.SArray(d);
				db->Dot(-learningRate);
				b->Add(db);
				delete db;
				delete d;

				return this->BPropH(pRes,dzPrev,i-1);
			}
			else{
				return dz;
			}
		}
		GpuArray<double> *DzFirst(GpuArray<double> *y,GpuArray<double> *yHat){
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *res,*t1,*t2,*t3;
			GpuVector<int> *dim = y->GetDim();
			int columns = dim->Get(1);
			
			t1 = op.Div(yHat,y);
			t2 = op.Dot(y,-1.0);
			t2->Add(1.0);
			t3 = op.Dot(yHat,-1.0);
			t3->Add(1.0);
			t3->Div(t2);
			res = op.Sub(t3,t1);
			res->Div(columns);

			delete t1;
			delete t2;
			delete t3;

			return res;
		}
		GpuArray<double> *TrainOneBProp(GpuArray<double> *input,GpuArray<double> *yHat){
			GVector < GpuArray <double> > *pRes = this->FPropB(input);
			GpuArray<double> *y = pRes->Get(layers - 1);
			GpuArray<double> *dz = this->DzFirst(y,yHat);
			GpuArray<double> *dzPrev = this->BPropH(pRes,dz,layers -1);

			delete pRes;

			return dzPrev;
		}
		void TrainBProp(GVector <GpuArray <double> > *input,GVector <GpuArray <double> > *yHat,int epohs){
			int baches = input->GetSize();
			GpuArray<double> *inpt,*yH,*temp;

			for (int i=0; i<baches; i++){
				inpt = input->Get(i);
				yH = yHat->Get(i);
				for (int j=0; j<epohs; j++){
					std::cout<<this->Loss(inpt,yH)<<"\n";
					temp = TrainOneBProp(inpt,yH);
					delete temp;
				}
			}
		}

};

class VFunc{
	int n; //Independent variables
	int m; //Max power
	int o; //number of outputs
	int blocks;
	int threads;
	double learningRate;
	GpuArray<double> *weights;
	GpuArray<double> *offsets;
	public:
		VFunc(){
			n = -1;
			m = -1;
			o = -1;
			blocks = -1;
			threads = -1;
			learningRate = -1;
			weights = NULL;
			offsets = NULL;
		}
		VFunc(int tN,int tM,int tO,int tBlocks,int tThreads){
			n = tN;
			m = tM;
			o = tO;
			blocks = tBlocks;
			threads = tThreads;
			learningRate = 0.001;
			GpuArrayOp<double> op(blocks,threads);
			weights = op.Rnd(2,o,n*m,0.9,1.1);
			offsets = op.Rnd(2,o,1,-1.0,1.0);
		}
		void Allocate(int tN,int tM,int tO,int tBlocks,int tThreads){
			n = tN;
			m = tM;
			o = tO;
			blocks = tBlocks;
			threads = tThreads;
			learningRate = 0.001;
			GpuArrayOp<double> op(blocks,threads);
			weights = op.Rnd(2,o,n*m,-1.0,1.0);
			offsets = op.Rnd(2,o,1,-1.0,1.0);
		}
		~VFunc(){
			delete weights;
			delete offsets;
		}
		void PrntWeights(){
			weights->Prnt();
		}
		GpuArray<double> * Out(GpuArray<double> *input){
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *rInput = op.Power(input,m);
			GpuArray<double> *t1 = op.Mul(weights,rInput);
			delete rInput;
			t1->Add(offsets);

			return t1;
		}
		double Loss(GpuArray<double> *input,GpuArray<double> *yHat){
			GpuArray<double> *y = this->Out(input);
			y->Sub(yHat);
			y->Pw(2,blocks,threads);
			GpuVector<int> *dim = y->GetDim();
			int columns = dim->Get(1);
			double res = y->Sum()/columns;
			delete y;

			return res;
		}
		void TrainOne(GpuArray<double> *input,GpuArray<double> *yHat){
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *y = this->Out(input);
			GpuVector<int> *dim = yHat->GetDim();
			int columns = dim->Get(1);
			
			y->Sub(yHat);
			y->Dot(2.0/columns);
			GpuArray<double> *rInput = op.Power(input,m);
			GpuArray<double> *inputTr = op.Transpose(rInput);
			GpuArray<double> *dw = op.Mul(y,inputTr);
			GpuArray<double> *db = op.SArray(y);
			
			dw->Dot(-learningRate);
			db->Dot(-learningRate);
			weights->Add(dw);
			offsets->Add(db);

			delete y;
			delete rInput;
			delete inputTr;
			delete dw;
			delete db;
		}
		void Train(GpuArray<double> *input,GpuArray<double> *yHat,int epohs){
			for (int i=0; i<epohs; i++){
				double loss = this->Loss(input,yHat);
				std::cout<<loss<<"\n";
				this->TrainOne(input,yHat);
			}
		}
};
/*
class VFuncFourier{
	int n; //Independent variables
	int m; //Max frequency
	int o; //number of outputs
	int blocks;
	int threads;
	double learningRate;
	GpuArray<double> *weights;
	GpuArray<double> *offsets;
	public:
		VFunc(){
			n = -1;
			m = -1;
			o = -1;
			blocks = -1;
			threads = -1;
			learningRate = -1;
			weights = NULL;
			offsets = NULL;
		}
		VFunc(int tN,int tM,int tO,int tBlocks,int tThreads){
			n = tN;
			m = tM;
			o = tO;
			blocks = tBlocks;
			threads = tThreads;
			learningRate = 0.001;
			GpuArrayOp<double> op(blocks,threads);
			weights = op.Rnd(2,o,n*m,0.9,1.1);
			offsets = op.Rnd(2,o,1,-1.0,1.0);
		}
		void Allocate(int tN,int tM,int tO,int tBlocks,int tThreads){
			n = tN;
			m = tM;
			o = tO;
			blocks = tBlocks;
			threads = tThreads;
			learningRate = 0.001;
			GpuArrayOp<double> op(blocks,threads);
			weights = op.Rnd(2,o,n*m,-1.0,1.0);
			offsets = op.Rnd(2,o,1,-1.0,1.0);
		}
		~VFunc(){
			delete weights;
			delete offsets;
		}
		void PrntWeights(){
			weights->Prnt();
		}
		GpuArray<double> * Out(GpuArray<double> *input){
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *rInput = op.Power(input,m);
			GpuArray<double> *t1 = op.Mul(weights,rInput);
			delete rInput;
			t1->Add(offsets);

			return t1;
		}
		double Loss(GpuArray<double> *input,GpuArray<double> *yHat){
			GpuArray<double> *y = this->Out(input);
			y->Sub(yHat);
			y->Pw(2,blocks,threads);
			GpuVector<int> *dim = y->GetDim();
			int columns = dim->Get(1);
			double res = y->Sum()/columns;
			delete y;

			return res;
		}
		void TrainOne(GpuArray<double> *input,GpuArray<double> *yHat){
			GpuArrayOp<double> op(blocks,threads);
			GpuArray<double> *y = this->Out(input);
			GpuVector<int> *dim = yHat->GetDim();
			int columns = dim->Get(1);
			
			y->Sub(yHat);
			y->Dot(2.0/columns);
			GpuArray<double> *rInput = op.Power(input,m);
			GpuArray<double> *inputTr = op.Transpose(rInput);
			GpuArray<double> *dw = op.Mul(y,inputTr);
			GpuArray<double> *db = op.SArray(y);
			
			dw->Dot(-learningRate);
			db->Dot(-learningRate);
			weights->Add(dw);
			offsets->Add(db);

			delete y;
			delete rInput;
			delete inputTr;
			delete dw;
			delete db;
		}
		void Train(GpuArray<double> *input,GpuArray<double> *yHat,int epohs){
			for (int i=0; i<epohs; i++){
				double loss = this->Loss(input,yHat);
				std::cout<<loss<<"\n";
				this->TrainOne(input,yHat);
			}
		}
};
*/

//
//Lexer
//

class LexerNode{
	public:
		virtual void Init(int tStartPos) = 0;
		virtual void Next(const char *str) = 0;
		virtual bool Pending() = 0;
		virtual int GetStopPos() = 0;
};

class LexerNodeLeaf: public LexerNode{
	int startPos;
	int currPos;
	int stopPos;
	int state;
	int goal;
	bool pending;
	CpuArray<int> transitionMatrix;
	public:
		LexerNodeLeaf(){
			startPos = 0;
			currPos = 0;
			stopPos = -1;
			state = 0;
			pending = true;
		}
		~LexerNodeLeaf(){}
		void And(const char *str){
			int length = strlen(str);
			CpuVector<int> *dim = new CpuVector<int>(2);
			dim->Set(0,length);
			dim->Set(1,128);
			transitionMatrix.Allocate(dim);
			transitionMatrix.Init(-1);

			for (int i=0; i<length; i++){
				transitionMatrix.Set(i,str[i],i+1);
			}

			goal = length;
		}
		void Or(const char *str){
			int length = strlen(str);
			CpuVector<int> *dim = new CpuVector<int>(2);
			dim->Set(0,1);
			dim->Set(1,128);
			transitionMatrix.Allocate(dim);
			transitionMatrix.Init(-1);

			for (int i=0; i<length; i++){
				transitionMatrix.Set(0,str[i],1);
			}

			goal = 1;

		}
		void Init(int tStartPos){
			startPos = tStartPos;
			currPos = tStartPos;
			stopPos = -1;
			state = 0;
			pending = true;
		}
		void Next(const char *str){
			if (pending){
				state = transitionMatrix.Get(state,str[currPos]);
				if (state == goal){
					pending = false;
					stopPos = currPos;
				}
				else if (state == -1){
					pending = false;
				}
				else{
					currPos++;
				}
			}
		}
		void TransitionMatrixPrnt(){
			transitionMatrix.Prnt();
		}
		bool Pending(){
			return pending;
		}
		int GetStopPos(){
			return stopPos;
		}
};

class LexerNodeStar:public LexerNode{
	int startPos;
	int stopPos;
	bool pending;
	LexerNode *lNode; 
	public:
		LexerNodeStar(LexerNode *tLNode){
			startPos = 0;
			stopPos = -1;
			pending = true;
			lNode = tLNode;
		}
		~LexerNodeStar(){
			delete lNode;
		}
		void Init(int tStartPos){
			startPos = tStartPos;
			stopPos = -1;
			pending = true;
			lNode->Init(tStartPos);
		}
		void Next(const char *str){
			if (pending){
				lNode->Next(str);
				if (!lNode->Pending()){
					int stPos;
					stPos = lNode->GetStopPos();
					if (stPos == -1){
						pending = false;
					}
					else{
						stopPos = stPos;
						lNode->Init(stPos + 1);
					}
				}
			}
		}
		bool Pending(){
			return pending;
		}
		int GetStopPos(){
			return stopPos;
		}
};

class LexerNodeAnd:public LexerNode{
	int startPos;
	int stopPos;
	bool pending;
	LListNode<LexerNode> *currNode;
	LList<LexerNode> nodeList;
	public:
		LexerNodeAnd(){
			startPos = 0;
			stopPos = -1;
			pending = true;
		}
		~LexerNodeAnd(){}
		void Add(LexerNode *nNode){
			nodeList.Add(nNode);
			currNode = nodeList.GetHead();
		}
		void Init(int tStartPos){
			LListNode<LexerNode> *temp = nodeList.GetHead();
			LexerNode *lNode;

			startPos = tStartPos;
			stopPos = -1;
			pending = true;
			currNode = nodeList.GetHead();

			while(temp != NULL){
				lNode = temp->GetData();
				lNode->Init(tStartPos);			
				temp = temp->GetNext();
			}
		}
		void Next(const char *str){
			if (pending){
				LexerNode *lNode = currNode->GetData();
				lNode->Next(str);
				if (!lNode->Pending()){
					int stPos = lNode->GetStopPos();
					if (stPos == -1){
						pending = false;
					}
					else{
						currNode = currNode->GetNext();
						if (currNode == NULL){
							pending = false;
							stopPos = stPos;
						}
						else{
							LexerNode *lNode = currNode->GetData();
							lNode->Init(stPos + 1);
						}
					}
				}
			}
		}
		bool Pending(){
			return pending;
		}
		int GetStopPos(){
			return stopPos;
		}

};

class LexerNodeOr:public LexerNode{
	int startPos;
	int stopPos;
	bool pending;
	LList<LexerNode> nodeList;
	public:
		LexerNodeOr(){
			startPos = 0;
			stopPos = -1;
			pending = true;
		}
		~LexerNodeOr(){}
		void Add(LexerNode *nNode){
			nodeList.Add(nNode);
		}
		void Init(int tStartPos){
			LListNode<LexerNode> *temp = nodeList.GetHead();
			LexerNode *lNode;

			startPos = tStartPos;
			stopPos = -1;
			pending = true;

			while(temp != NULL){
				lNode = temp->GetData();
				lNode->Init(tStartPos);			
				temp = temp->GetNext();
			}
		}
		void Next(const char *str){
			if (pending){
				LListNode<LexerNode> *temp = nodeList.GetHead();
				LexerNode *lNode;
				bool pndng = false;
				int stPos = -1;

				while(temp != NULL){
					lNode = temp->GetData();
					lNode->Next(str);
					bool nodePending = lNode->Pending();
					if (!nodePending){
						if (lNode->GetStopPos() > stPos){
							stPos = lNode->GetStopPos();
						}
					}
					pndng = pndng || nodePending;
					temp = temp->GetNext();
				}

				if (!pndng){
					pending = false;
					stopPos = stPos;
				}
			}
		}
		bool Pending(){
			return pending;
		}
		int GetStopPos(){
			return stopPos;
		}
};

class LexerNodeRoot:public LexerNode{
	int startPos;
	int stopPos;
	bool pending;
	int token;
	LList<LexerNode> nodeList;
	public:
		LexerNodeRoot(){
			startPos = 0;
			stopPos = -1;
			pending = true;
			token = -1;
		}
		~LexerNodeRoot(){}
		void Add(LexerNode *nNode){
			nodeList.Add(nNode);
		}
		void Init(int tStartPos){
			LListNode<LexerNode> *temp = nodeList.GetHead();
			LexerNode *lNode;

			startPos = tStartPos;
			stopPos = -1;
			pending = true;

			while(temp != NULL){
				lNode = temp->GetData();
				lNode->Init(tStartPos);			
				temp = temp->GetNext();
			}
		}
		void Next(const char *str){
			if (pending){
				LListNode<LexerNode> *temp = nodeList.GetHead();
				LexerNode *lNode;
				bool pndng = false;
				int stPos = -1;
				int iter = 0;
				int tokenId = -1;

				while(temp != NULL){
					lNode = temp->GetData();
					lNode->Next(str);
					bool nodePending = lNode->Pending();
					if (!nodePending){
						if (lNode->GetStopPos() > stPos){
							tokenId = iter;
							stPos = lNode->GetStopPos();
						}
					}
					pndng = pndng || nodePending;
					temp = temp->GetNext();
					iter++;
				}

				if (!pndng){
					pending = false;
					stopPos = stPos;
					token = tokenId;
				}
			}
		}
		bool Pending(){
			return pending;
		}
		int GetStopPos(){
			return stopPos;
		}
		int GetToken(){
			return token;
		}
};

class LexerToken{
	int tokenNum;	//-1 Error,-2 EOS
	int length;
	char *data;
	public:
		LexerToken(int tTokenNum,const char *str,int start,int stop){
			tokenNum = tTokenNum;
			length = stop - start + 1;
			data = (char *)malloc(length * sizeof(char));

			for (int i=0; i<length; i++){
				data[i] = str[start + i];
			}
		}
		LexerToken(int tTokenNum){
			tokenNum = tTokenNum;
			length = 0;
			data = NULL;
		}
		~LexerToken(){
			free(data);
		}
		void PrntData(){
			printf("%s",data);
		}
		void PrntToken(){
			printf("---Token---\n");
			printf("TokenNum = %d\n",tokenNum);
			printf("Data = %s\n",data);
		}
		int GetTokenNum(){
			return tokenNum;
		}
		int GetLength(){
			return length;
		}
		char *GetData(){
			return data;
		}
		void IncToken(int offset){
			if (tokenNum >= 0){
				tokenNum = tokenNum + offset;
			}
		}

};

class Lexer{
	LexerNodeRoot *root;
	int startPos;
	char *str;
	int strLength;
	public:
		Lexer(LexerNodeRoot *tRoot){
			root = tRoot;
			startPos = 0;
			str = NULL;
			strLength = 0;
		}
		~Lexer(){
			delete root;
			free(str);
		}
		void AddString(const char *tStr){
			strLength = strlen(tStr);
			str = (char *)malloc(strLength * sizeof(char));

			for (int i=0; i<strLength; i++){
				str[i] = tStr[i];
			}
		}
		LexerToken *GetNext(){
			LexerToken *res;

			if (startPos < strLength){
				while(root->Pending()){
					root->Next(str);
				}		
				int stopPos = root->GetStopPos();
				if (stopPos == -1){
					res = new LexerToken(-1);
					startPos++;
				}
				else{
					res = new LexerToken(root->GetToken()
							,str,startPos,stopPos);
					startPos = stopPos + 1;
				}
				root->Init(startPos);
			}
			else{
				res = new LexerToken(-2);
			}

			return res;

		}
};

class PrntMatrixPoint{
	int x;
	int y;
	int manhatan;
	PrntMatrixPoint *next;
	public:
		PrntMatrixPoint(int tX,int tY,int x2,int y2){
			x = tX;
			y = tY;
			manhatan = this->Manhatan(x,y,x2,y2);
			next = NULL;
		}
		~PrntMatrixPoint(){}
		int Abs(int x1){
			if (x1<0){
				return -x1;
			}
			return x1;
		}
		int Manhatan(int x1,int y1,int x2,int y2){
			return this->Abs(x1-x2) + this->Abs(y1-y2);
		}
		void SetNext(PrntMatrixPoint *nPoint){
			next = nPoint;
		}
		PrntMatrixPoint *GetNext(){
			return next;
		}
		int GetManhatan(){
			return manhatan;
		}
		int GetX(){
			return x;
		}
		int GetY(){
			return y;
		}
};

class PrntMatrixPointLst{
	PrntMatrixPoint *head;
	int xDest;
	int yDest;
	bool reachedDest;
	public:
		PrntMatrixPointLst(int tXDest,int tYDest){
			xDest = tXDest;
			yDest = tYDest;
			head = NULL;
			reachedDest = false;
		}
		~PrntMatrixPointLst(){
			PrntMatrixPoint *temp = head;
			if (temp != NULL){
				PrntMatrixPoint *temp2 = temp->GetNext();
				while(temp2 != NULL){
					delete temp;
					temp = temp2;
					temp2 = temp2->GetNext();
				}
				delete temp;
			}
		}
		void Add(int x,int y){
			PrntMatrixPoint *nPoint = new PrntMatrixPoint(x,y,xDest,yDest);
			int mnh1 = nPoint->GetManhatan();
			if (mnh1 == 0){
				reachedDest = true;
			}
			int mnh2;
			PrntMatrixPoint *temp = head;
			if (temp != NULL){
				mnh2 = temp->GetManhatan();
				if (mnh1 < mnh2){
					nPoint->SetNext(head);
					head = nPoint;
				}
				else{
					PrntMatrixPoint *temp2 = temp->GetNext();
					while(temp2 != NULL){
						mnh2 = temp2->GetManhatan();
						if (mnh1 < mnh2){
							nPoint->SetNext(temp2);
							temp->SetNext(nPoint);
							break;
						}
						temp = temp2;
						temp2 = temp2->GetNext();
					}
					if (temp2 == NULL){
						temp->SetNext(nPoint);
					}
				}

			}
			else{
				head = nPoint;
			}
		}
		PrntMatrixPoint *GetHead(){
			return head;
		}
		bool ReachedDest(){
			return reachedDest;
		}
};

class PrntMatrix{
	int rows;
	int columns;
	CpuArray<char> prntMatrix;
	CpuVector<int> lastCharPosHrz;	
	public:
		PrntMatrix(int tRows,int tColumns){
			rows = tRows;
			columns = tColumns;
			lastCharPosHrz.Allocate(rows);
			lastCharPosHrz.Init(0);
			CpuVector<int> *dim = new CpuVector<int>(2);
			dim->Set(0,rows);
			dim->Set(1,columns);
			prntMatrix.Allocate(dim);
		}
		~PrntMatrix(){}
		void Init(char c){
			prntMatrix.Init(c);
		}
		int PrntH(int r,const char *str,int xConnect,int yConnect){
			int strLength = strlen(str);
			int start = lastCharPosHrz.Get(r) + 4;
			this->Connect(r,start,xConnect,yConnect);
			lastCharPosHrz.Set(r,start + strLength);
			CpuVector<int> *temp;
			for (int i=0; i<strLength; i++){
				temp = new CpuVector<int>(2);
				temp->Set(0,r);
				temp->Set(1,start+i);
				prntMatrix.Set(temp,str[i]);
				delete temp;
			}
			return start;
		}
		void PrntHorizontalLine(int startColumn,int stopColumn,int row){
			CpuVector<int> *temp;
			for (int i=startColumn; i<stopColumn; i++){
				temp = new CpuVector<int>(2);
				temp->Set(0,row);
				temp->Set(1,i);
				prntMatrix.Set(temp,'-');
				delete temp;
			}
		}
		void Prnt(){
			for (int i=0; i<rows; i++){
				for (int j=0; j<columns; j++){
					printf("%c",prntMatrix.Get(i,j));
				}
				printf("\n");
			}
		}
		void Connect(int x1,int y1,int x2,int y2){
			PrntMatrixPointLst l(x2,y2);
			l.Add(x1-1,y1-1);
			l.Add(x1-1,y1);
			l.Add(x1-1,y1+1);
			l.Add(x1,y1-1);
			l.Add(x1,y1);
			l.Add(x1,y1+1);
			l.Add(x1+1,y1-1);
			l.Add(x1+1,y1);
			l.Add(x1+1,y1+1);
			if (!l.ReachedDest()){
				PrntMatrixPoint *temp = l.GetHead();
				int x;
				int y;
				while (temp != NULL){
					x = temp->GetX();
					y = temp->GetY();	
					if (this->InTheMap(x,y)){
						CpuVector<int> *t;
						t = new CpuVector<int>(2);
						t->Set(0,x);
						t->Set(1,y);
						if (prntMatrix.Get(x,y) == ' '){
							prntMatrix.Set(t,'*');
						}
						delete t;
						break;
					}
					temp = temp->GetNext();
				}
				if (temp != NULL){
					this->Connect(x,y,x2,y2);
				}
			}
		}
		bool InTheMap(int x1,int y1){
			return ((0 <= x1) && (x1 <rows) && (0 <= y1) && (y1 <columns));
		}
		bool ValidMove(int x1,int y1){
			return (this->InTheMap(x1,y1) && (prntMatrix.Get(x1,y1) == ' '));
		}
		int Manhatan(int x1,int y1,int x2,int y2){
			int d1,d2;
			if (x1 >= x2){
				d1 = x1 - x2;
			}
			else{
				d1 = x2 - x1;
			}
			if (y1 >= y2){
				d2 = y1 - y2;
			}
			else{
				d2 = y2 - y1;
			}
			return d1 + d2;
		}
};

class ASTNode{
	int id;
	char *name;
	bool isTerminal;
	char *data;
	int level;	//0 for root
	ASTNode *anc;
	LList<ASTNode> subtree;
	public:
		ASTNode(){
			name = NULL;
			data = NULL;
			isTerminal = false;
			anc = NULL;
		}
		~ASTNode(){
			free(name);
			free(data);
		}
		void Add(ASTNode *nNode){
			nNode->SetAnc(this);
			subtree.Add(nNode);
		}
		void SetName(const char *tName){
			int nameLength = strlen(tName);
			name = (char *)malloc(nameLength * sizeof(char));
			for (int i=0; i<nameLength; i++){
				name[i] = tName[i];
			}
		}
		void SetTerminal(){
			isTerminal = true;
		}
		void SetData(const char *tData){
			int dataLength = strlen(tData);
			data = (char *)malloc(dataLength * sizeof(char));
			for (int i=0; i<dataLength; i++){
				data[i] = tData[i];
			}
		}
		void SetId(int tId){
			id = tId;
		}
		int GetId(){
			return id;
		}
		char *GetName(){
			return name;
		}
		bool IsTerminal(){
			return isTerminal;
		}
		char *GetData(){
			return data;
		}
		void PrntH(PrntMatrix *prntMatrix,int r,int xConnect,int yConnect){
			int yStart;
			yStart = prntMatrix->PrntH(r,name,xConnect,yConnect);
			LListNode<ASTNode> *temp = subtree.GetHead();
			ASTNode *astNode;
			while(temp != NULL){
				astNode = temp->GetData();
				astNode->PrntH(prntMatrix,r + 10,r+1,yStart);
				temp = temp->GetNext();
			}
		}
		void Prnt(){
			PrntMatrix *prntMatrix = new PrntMatrix(200,200);
			prntMatrix->Init(' ');
			this->PrntH(prntMatrix,0,0,0);
			prntMatrix->Prnt();
			delete prntMatrix;
		}
		int GetNumOfSubNodes(){
			if (!isTerminal){
				return subtree.GetNumOfNodes();
			}
			return 0;
		}
		void InitLevel(){
			if (anc == NULL){
				level = 0;
				LListNode<ASTNode> *temp = subtree.GetHead();
				ASTNode *astNode;
				while(temp != NULL){
					astNode = temp->GetData();
					astNode->InitLevel();
					temp = temp->GetNext();
				}
			}
			else{
				level = anc->GetLevel() + 1;
				LListNode<ASTNode> *temp = subtree.GetHead();
				ASTNode *astNode;
				while(temp != NULL){
					astNode = temp->GetData();
					astNode->InitLevel();
					temp = temp->GetNext();
				}
			}
		}
		int GetLevel(){
			return level;
		}
		void SetAnc(ASTNode *tAnc){
			anc = tAnc;
		}
		LListNode<ASTNode> *GetHead(){
			return subtree.GetHead();
		}
};


template <class T>
class LifoNode{
	T *data;
	LifoNode *prev;
	bool deleteMode;	//if deleteMode = true then data its deleted
				//in the destructor
	public:
		LifoNode(T *tData){
			data = tData;
			prev = NULL;
			deleteMode = false;
		}
		~LifoNode(){
			if(deleteMode){
				delete data;
			}
		}
		T *GetData(){
			return data;
		}
		void SetPrev(LifoNode *tPrev){
			prev = tPrev;
		}
		LifoNode *GetPrev(){
			return prev;
		}
		void SetDeleteMode(){
			deleteMode = true;
		}
};

template <class T>
class Lifo{
	LifoNode<T> *head;
	public:
		Lifo(){
			head = NULL;
		}
		~Lifo(){
			LifoNode<T> *temp = head;
			if (temp != NULL){
				LifoNode<T> *temp2 = temp->GetPrev();
				while(temp2 != NULL){
					temp->SetDeleteMode(); //Osa emeinan del
					delete temp;
					temp = temp2;
					temp2 = temp2->GetPrev();
				}
				delete temp;
			}
		}
		void Push(T *data){
			LifoNode<T> *nNode = new LifoNode<T>(data);
			nNode->SetPrev(head);
			head = nNode;
		}
		T *Pop(){
			T *data;
			if (head == NULL){
				data = NULL;
			}
			else{
				data = head->GetData();
				LifoNode<T> *temp = head;
				head = head->GetPrev();
				delete temp;
			}
			return data;
		}
		void Clear(){
			LifoNode<T> *temp = head;
			if (temp != NULL){
				LifoNode<T> *temp2 = temp->GetPrev();
				while(temp2 != NULL){
					temp->SetDeleteMode(); 
					delete temp;
					temp = temp2;
					temp2 = temp2->GetPrev();
				}
				delete temp;
			}
			head = NULL;
				
		}
};

class SingleProduction{
	LList<int> l;
	public:
		SingleProduction(){
		}
		~SingleProduction(){
		}
		void Add(int t){
			int *a = (int *)malloc(sizeof(int));
			*a = t;
			l.Add(a);
		}
		//returns a pointer to LListNode with data t
		LListNode<int> *Find(int t){
			LListNode<int> *temp = l.GetHead();
			int *a;
			LListNode<int> *res = NULL;
			while(temp != NULL){
				a = temp->GetData();
				if (*a == t){
					res = temp;
					break;
				}
				temp = temp->GetNext();
			}
			return res;
		}
		LListNode<int> *GetHead(){
			return l.GetHead();
		}
};

class NTerminalProd{
	int id;
	LList<SingleProduction> prod;
	public:
		NTerminalProd(){
		}
		NTerminalProd(int tId){
			id = tId;
		}
		~NTerminalProd(){}
		void SetId(int tId){
			id = tId;
		}
		void Add(SingleProduction *nProd){
			prod.Add(nProd);
		}
		int GetId(){
			return id;
		}
		LListNode<SingleProduction> *GetHead(){
			return prod.GetHead();
		}
};

class SingleProductionR{
	int nTerminalId;
	LListNode<int> *currentNode;
	LList<int> l;
	bool pending;
	public:
		SingleProductionR(int tNTerminalId,SingleProduction *sP){
			nTerminalId = tNTerminalId;
			l.SetRMode();	//Set Reverce mode
			pending = true;
			LListNode<int> *temp = sP->GetHead();
			while(temp != NULL){
				int *a = temp->GetData();
				int t = *a;
				this->Add(t);
				temp = temp->GetNext();
			}
		}
		~SingleProductionR(){}
		void Add(int t){
			int *a = (int *)malloc(sizeof(int));
			*a = t;
			l.Add(a);
		}
		int Next(int t){
			if ((pending) && (currentNode != NULL)){
				int *a = currentNode->GetData();
				LListNode<int> *nxt = currentNode->GetNext();
				if (*a == t){
					if (nxt == NULL){
						pending = false;
						return nTerminalId;
					}
					else{
						currentNode = nxt;
					}
				}
				else{
					pending = false;
				}
			}
			return -1;
		}
		void Init(){
			currentNode = l.GetHead();
			pending = true;
		}
		bool Pending(){
			return pending;
		}
		int GetLength(){
			return l.GetNumOfNodes();
		}
};

class ReduceStruct{
	LList<SingleProductionR> l;
	bool pending;
	int length;	//Length of the production
	int id;		//id of the nTerminal
	public:
		ReduceStruct(){
			pending = true;
			length = 0;
			id = -1;
		}
		~ReduceStruct(){}
		void Add(SingleProductionR *s){
			l.Add(s);
		}
		void Init(){
			pending = true;
			length = 0;
			id = -1;
			LListNode<SingleProductionR> *temp = l.GetHead();
			while(temp != NULL){
				SingleProductionR *s = temp->GetData();
				s->Init();
				temp = temp->GetNext();
			}
		}
		void Next(int t){
			LListNode<SingleProductionR> *temp = l.GetHead();
			bool pndng = false;
			while(temp != NULL){
				SingleProductionR *s = temp->GetData();
				int a = s->Next(t);
				pndng = (pndng || s->Pending());
				if (a != -1){
					id = a;
					length = s->GetLength();
				}
				temp = temp->GetNext();
			}
			pending = pndng;
		}
		int GetLength(){
			return length;
		}
		bool Pending(){
			return pending;
		}
		int GetId(){
			return id;
		}
};

//Abstract Syntax Tree Generator
class ASTGen{
	int all;	//number of terminals + not terminals
	int nTerminals; //number of not Terminals
	GVector<char> names; //0->"S",1->"Expr",...
	GVector<NTerminalProd> grammar;
	CpuArray<bool> firstSet;
	CpuVector<bool> firstSetCompleted;
	CpuArray<bool> followSet;
	ReduceStruct rStruct;
	int lastSymbolId; //last symbol id in the lifo
	Lifo<ASTNode> lifo;
	Lexer *lexer;
	public:
		ASTGen(){}
		ASTGen(int tAll,int tNTerminals){
			this->Allocate(tAll,tNTerminals);
		}
		void Allocate(int tAll,int tNTerminals){
			all = tAll;
			nTerminals = tNTerminals;
			names.Allocate(all);
			grammar.Allocate(nTerminals);
			NTerminalProd *nTermProd;
			for (int i=0; i<nTerminals; i++){
				nTermProd = new NTerminalProd(i);
				grammar(i,nTermProd);
			}
			CpuVector<int> *dim = new CpuVector<int>(2);
			dim->Set(0,all);
			dim->Set(1,all);
			firstSet.Allocate(dim);
			firstSet.Init(false);
			CpuVector<int> *temp;
			firstSetCompleted.Allocate(all);
			firstSetCompleted.Init(false);
			for (int i=nTerminals; i<all; i++){
				temp = new CpuVector<int>(2);
				temp->Set(0,i);
				temp->Set(1,i);
				firstSet.Set(temp,true);
				delete temp;
				firstSetCompleted.Set(i,true);
			}
			CpuVector<int> *followDim = new CpuVector<int>(2);
			followDim->Set(0,all);
			followDim->Set(1,all);
			followSet.Allocate(followDim);
			followSet.Init(false);
		}
		void PrntFirstSet(){
			firstSet.Prnt();
		}
		void PrntFollowSet(){
			followSet.Prnt();
		}
		~ASTGen(){
			delete lexer;
		}
		void SetName(int id,const char *tName){
			int nameLength = strlen(tName);
			char *name = (char *)malloc(nameLength * sizeof(char));
			for (int i=0; i<nameLength; i++){
				name[i] = tName[i];
			}
			names(id,name);
		}
		void Add(int id,SingleProduction *prod){
			NTerminalProd *nTerm = grammar.Get(id);
			nTerm->Add(prod);
		}
		bool IsTerminal(int t){
			return t>=nTerminals;
		}
		//line toBeOred = toBeOred | src
		void FirstSetOr(int toBeOred,int src){
			CpuVector<int> *temp;
			bool val;
			for (int i=0; i<all; i++){
				val = firstSet.Get(toBeOred,i) | firstSet.Get(src,i);
				temp = new CpuVector<int>(2);
				temp->Set(0,toBeOred);
				temp->Set(1,i);
				firstSet.Set(temp,val);
				delete temp;
			}
		}
		void FollowSetOr(int toBeOred,int src){
			CpuVector<int> *temp;
			bool val;
			for (int i=0; i<all; i++){
				val = followSet.Get(toBeOred,i) | firstSet.Get(src,i);
				temp = new CpuVector<int>(2);
				temp->Set(0,toBeOred);
				temp->Set(1,i);
				followSet.Set(temp,val);
				delete temp;
			}
		}
		void ComputeFirstSetH(int id){
			if (!firstSetCompleted.Get(id)){
				NTerminalProd *prod = grammar.Get(id);
				LListNode<SingleProduction> *temp = prod->GetHead();
				while(temp != NULL){
					SingleProduction *sP = temp->GetData();
					LListNode<int> *fNum = sP->GetHead();
					int *a = fNum->GetData();
					if (*a != id){
						this->ComputeFirstSetH(*a);
						this->FirstSetOr(id,*a);
					}
					temp = temp->GetNext();
				}
				firstSetCompleted.Set(id,true);
			}
		}
		void ComputeFirstSet(){
			for (int i=0; i<nTerminals; i++){
				this->ComputeFirstSetH(i);
			}
		}
		void ComputeFollowSetH(int sId){
			for (int i=0; i<nTerminals; i++){
				NTerminalProd *prod = grammar.Get(i);
				LListNode<SingleProduction> *temp = prod->GetHead();
				while(temp != NULL){
					SingleProduction *sP = temp->GetData();
					LListNode<int> *s,*t;
					s = sP->GetHead();
					while(s != NULL){
						int *syntaxNode = s->GetData();
						if (*syntaxNode == sId){
							t = s->GetNext();
							if (t != NULL){
								int *tData = t->GetData();
								this->FollowSetOr(sId,*tData);
							}
						}	
						s = s->GetNext();
					}
					temp = temp->GetNext();
				}
			}
		}
		//The first set must be constructed before ComputeFollowSet
		void ComputeFollowSet(){
			for (int i=0; i<all; i++){
				this->ComputeFollowSetH(i);
			}
		}
		//The grammar must be constucted before calling ConstructRStruct
		void ConstructRStruct(){
			for (int i=0; i<nTerminals; i++){
				NTerminalProd *prod = grammar.Get(i);
				LListNode<SingleProduction> *temp = prod->GetHead();
				while (temp != NULL){
					SingleProduction *sP = temp->GetData();
					SingleProductionR *sPR = new SingleProductionR(i,sP);
					rStruct.Add(sPR);
					temp = temp->GetNext();
				}
			}
		}
		//Returns true if it found a reduction
		bool Reduce(){
			rStruct.Init();
			bool res = false;
			ASTNode *nNode = new ASTNode();
			ASTNode *temp = lifo.Pop();
			Lifo<ASTNode> tempLifo;
			int iter = 0;
			while((temp != NULL)&&(rStruct.Pending())){
				tempLifo.Push(temp);
				iter++;
				int t = temp->GetId();
				rStruct.Next(t);
				temp = lifo.Pop();
			}
			
			if (temp != NULL){
				lifo.Push(temp);
			}

			int id = rStruct.GetId();
			if (id != -1){
				int length = rStruct.GetLength();
				int pushBack = iter - length;
				for (int i=0; i<pushBack; i++){
					temp = tempLifo.Pop();
					lifo.Push(temp);
				}
				temp = tempLifo.Pop();
				while(temp != NULL){
					nNode->Add(temp);
					temp = tempLifo.Pop();
				}
				nNode->SetId(id);
				nNode->SetName(names.Get(id));
				lifo.Push(nNode);
				lastSymbolId = id;
				res = true;
			}
			return res;
		}
		ASTNode* GetAST(const char *str){
			lifo.Clear();
			lexer->AddString(str);
			LexerToken *t;
			t = lexer->GetNext();
			int tokenNum = t->GetTokenNum();
			if (tokenNum >= 0){
				ASTNode *nNode = new ASTNode();
				nNode->SetId(tokenNum + nTerminals);
				nNode->SetName(names.Get(tokenNum + nTerminals));
				nNode->SetTerminal();
				nNode->SetData(t->GetData());
				lastSymbolId = tokenNum + nTerminals;
				lifo.Push(nNode);
				delete t;
				t = lexer->GetNext();
				tokenNum = t->GetTokenNum();
			}
			else{
				return NULL;
			}
			while(tokenNum != -2){
				if (tokenNum == -1){
					printf("Token Error\n");
					return NULL;
				}
				else{
					if (followSet.Get(lastSymbolId,tokenNum + nTerminals)){
						ASTNode *nNode = new ASTNode();
						nNode->SetId(tokenNum + nTerminals);
						nNode->SetName(names.Get(tokenNum + nTerminals));
						nNode->SetTerminal();
						nNode->SetData(t->GetData());
						lastSymbolId = tokenNum + nTerminals;
						lifo.Push(nNode);
						delete t;
						t = lexer->GetNext();
						tokenNum = t->GetTokenNum();
					}
					else{
						if (!this->Reduce()){
							printf("Syntaax Error\n");
							return NULL;
						}
					}
				}

			}
			while (lastSymbolId != 0){
				printf("%d\n",lastSymbolId);
				if (!this->Reduce()){
					printf("Syntaxx Error\n");
					return NULL;
				}
			}
			ASTNode *res = lifo.Pop();

			return res;
		}
		void SetLexer(Lexer *tLexer){
			lexer = tLexer;
		}
};

class FOfSeVarNode{
	int id;
	char *name;
	FOfSeVarNode *left;
	FOfSeVarNode *right;
	FOfSeVarNode *anc;
	int varNum;
	int intVal;
	double doubleVal;
	public:
		FOfSeVarNode(){
			left = NULL;
			right = NULL;
			anc = NULL;
		}
		~FOfSeVarNode(){
			free(name);
			delete left;
			delete right;
		}
		void SetLeft(FOfSeVarNode *tLeft){
			left = tLeft;
			tLeft->SetAnc(this);
		}
		void SetRight(FOfSeVarNode *tRight){
			right = tRight;
			tRight->SetAnc(this);
		}
		void SetAnc(FOfSeVarNode *tAnc){
			anc = tAnc;
		}
		void SetVarNum(int tVarNum){
			varNum = tVarNum;
		}
		int GetVarNum(){
			return varNum;
		}
		void SetIntVal(int tIntVal){
			intVal = tIntVal;
		}
		int GetIntVal(){
			return intVal;
		}
		void SetDoubleVal(double tDoubleVal){
			doubleVal = tDoubleVal;
		}
		double GetDoubleVal(){
			return doubleVal;
		}
		void SetPlus(){
			id = 0;
			name = (char *)malloc(sizeof(strlen("PLUS")));
			strcpy(name,"PLUS");
		}
		void SetMinus(){
			id = 1;
			name = (char *)malloc(sizeof(strlen("MINUS")));
			strcpy(name,"MINUS");
		}
		void SetMull(){
			id = 2;
			name = (char *)malloc(sizeof(strlen("MULL")));
			strcpy(name,"MULL");
		}
		void SetDiv(){
			id = 3;
			name = (char *)malloc(sizeof(strlen("DIV")));
			strcpy(name,"DIV");
		}
		void SetInt(){
			id = 4;
			name = (char *)malloc(sizeof(strlen("INT")));
			strcpy(name,"INT");
		}
		void SetDouble(){
			id = 5;
			name = (char *)malloc(sizeof(strlen("DOUBLE")));
			strcpy(name,"DOUBLE");
		}
		void SetVar(){
			id = 6;
			name = (char *)malloc(sizeof(strlen("VAR")));
			strcpy(name,"VAR");
		}
		void SetSin(){
			id = 9;
			name = (char *)malloc(sizeof(strlen("SIN")));
			strcpy(name,"SIN");
		}
		void SetCos(){
			id = 10;
			name = (char *)malloc(sizeof(strlen("COS")));
			strcpy(name,"COS");
		}
		void SetTan(){
			id = 11;
			name = (char *)malloc(sizeof(strlen("TAN")));
			strcpy(name,"TAN");
		}
		void SetArcSin(){
			id = 12;
			name = (char *)malloc(sizeof(strlen("ARCSIN")));
			strcpy(name,"ARCSIN");
		}
		void SetArcCos(){
			id = 13;
			name = (char *)malloc(sizeof(strlen("ARCCOS")));
			strcpy(name,"ARCCOS");
		}
		void SetArcTan(){
			id = 14;
			name = (char *)malloc(sizeof(strlen("ARCTAN")));
			strcpy(name,"ARCTAN");
		}
		void SetLog(){
			id = 15;
			name = (char *)malloc(sizeof(strlen("LOG")));
			strcpy(name,"LOG");
		}
		void SetSqrt(){
			id = 16;
			name = (char *)malloc(sizeof(strlen("SQRT")));
			strcpy(name,"SQRT");
		}
		void SetExp(){
			id = 17;
			name = (char *)malloc(sizeof(strlen("EXP")));
			strcpy(name,"EXP");
		}
		void SetPow(){
			id = 18;
			name = (char *)malloc(sizeof(strlen("POW")));
			strcpy(name,"POW");
		}
		bool IsPlus(){
			return id==0;
		}
		bool IsMinus(){
			return id==1;
		}
		bool IsMull(){
			return id==2;
		}
		bool IsDiv(){
			return id==3;
		}
		bool IsInt(){
			return id==4;
		}
		bool IsDouble(){
			return id == 5;
		}
		bool IsVar(){
			return id == 6;
		}
		bool IsSin(){
			return id == 9;
		}
		bool IsCos(){
			return id == 10;
		}
		bool IsTan(){
			return id == 11;
		}
		bool IsArcSin(){
			return id == 12;
		}
		bool IsArcCos(){
			return id == 13;
		}
		bool IsArcTan(){
			return id == 14;
		}
		bool IsLog(){
			return id == 15;
		}
		bool IsSqrt(){
			return id == 16;
		}
		bool IsExp(){
			return id == 17;
		}
		bool IsPow(){
			return id == 18;
		}
		void PrntH(PrntMatrix *prntMatrix,int r,int xConnect,int yConnect){
			int yStart;
			yStart = prntMatrix->PrntH(r,name,xConnect,yConnect);
			if (left != NULL){
				left->PrntH(prntMatrix,r+10,r+1,yStart);
			}
			if (right != NULL){
				right->PrntH(prntMatrix,r+10,r+1,yStart);
			}
		}
		void Prnt(){
			PrntMatrix *prntMatrix = new PrntMatrix(200,200);
			prntMatrix->Init(' ');
			this->PrntH(prntMatrix,0,0,0);
			prntMatrix->Prnt();
			delete prntMatrix;
		}
		FOfSeVarNode *GetLeft(){
			return left;
		}
		FOfSeVarNode *GetRight(){
			return right;
		}
		FOfSeVarNode *Cpy(){
			if (this->IsPlus()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetPlus();
				FOfSeVarNode *l = left->Cpy();
				FOfSeVarNode *r = right->Cpy();
				res->SetLeft(l);
				res->SetRight(r);
				return res;
			}
			else if (this->IsMinus()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetMinus();
				FOfSeVarNode *l = left->Cpy();
				FOfSeVarNode *r = right->Cpy();
				res->SetLeft(l);
				res->SetRight(r);
				return res;
			}
			else if (this->IsMull()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetMull();
				FOfSeVarNode *l = left->Cpy();
				FOfSeVarNode *r = right->Cpy();
				res->SetLeft(l);
				res->SetRight(r);
				return res;
			}
			else if (this->IsDiv()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetDiv();
				FOfSeVarNode *l = left->Cpy();
				FOfSeVarNode *r = right->Cpy();
				res->SetLeft(l);
				res->SetRight(r);
				return res;
			}
			else if (this->IsInt()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetInt();
				res->SetIntVal(intVal);
				return res;
			}
			else if (this->IsDouble()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetDouble();
				res->SetDoubleVal(doubleVal);
				return res;
			}
			else if (this->IsVar()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetVar();
				res->SetVarNum(varNum);
				return res;
			}
			else if (this->IsSin()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetSin();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsCos()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetCos();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsTan()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetTan();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsArcSin()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcSin();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsArcCos()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcCos();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsArcTan()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcTan();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsLog()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetLog();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsSqrt()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetSqrt();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsExp()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetExp();
				FOfSeVarNode *l = left->Cpy();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsPow()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetPow();
				FOfSeVarNode *l = left->Cpy();
				FOfSeVarNode *r = right->Cpy();
				res->SetLeft(l);
				res->SetRight(r);
				return res;
			}
			return NULL;
		}
		FOfSeVarNode * PDer(int indVar){
			if (this->IsPlus()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetPlus();
				FOfSeVarNode *lDer = left->PDer(indVar);
				FOfSeVarNode *rDer = right->PDer(indVar);
				res->SetLeft(lDer);
				res->SetRight(rDer);

				return res;
			}
			else if (this->IsMinus()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetMinus();
				FOfSeVarNode *lDer = left->PDer(indVar);
				FOfSeVarNode *rDer = right->PDer(indVar);
				res->SetLeft(lDer);
				res->SetRight(rDer);

				return res;
			}
			else if (this->IsMull()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *headLeft = new FOfSeVarNode();
				FOfSeVarNode *headRight = new FOfSeVarNode();
				
				res->SetPlus();
				headLeft->SetMull();
				headRight->SetMull();
				res->SetLeft(headLeft);
				res->SetRight(headRight);
				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *rCpy = right->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);
				FOfSeVarNode *rDer = right->PDer(indVar);
				headLeft->SetLeft(lDer);
				headLeft->SetRight(rCpy);
				headRight->SetLeft(lCpy);
				headRight->SetRight(rDer);

				return res;
			}
			else if (this->IsDiv()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *minus = new FOfSeVarNode();
				FOfSeVarNode *mull0 = new FOfSeVarNode();
				FOfSeVarNode *mull1 = new FOfSeVarNode();
				FOfSeVarNode *tPow = new FOfSeVarNode();
				FOfSeVarNode *tow = new FOfSeVarNode();

				FOfSeVarNode *lDer = left->PDer(indVar);
				FOfSeVarNode *rCpy0 = right->Cpy();
				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *rDer = right->PDer(indVar);
				FOfSeVarNode *rCpy1 = right->Cpy();

				mull0->SetMull();
				mull0->SetLeft(lDer);
				mull0->SetRight(rCpy0);
				mull1->SetMull();
				mull1->SetLeft(lCpy);
				mull1->SetRight(rDer);
				minus->SetMinus();
				minus->SetLeft(mull0);
				minus->SetRight(mull1);
				tow->SetInt();
				tow->SetIntVal(2);
				tPow->SetPow();
				tPow->SetLeft(rCpy1);
				tPow->SetRight(tow);

				return res;
			}
			else if (this->IsInt()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetInt();
				res->SetIntVal(0);

				return res;
			}
			else if (this->IsDouble()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetDouble();
				res->SetDoubleVal(0);

				return res;
			}
			else if (this->IsVar()){
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetInt();

				if (varNum == indVar){
					res->SetIntVal(1);
				}
				else{
					res->SetIntVal(0);
				}

				return res;
			}
			else if (this->IsSin()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *tCos = new FOfSeVarNode();
				
				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				tCos->SetCos();
				tCos->SetLeft(lCpy);
				res->SetMull();
				res->SetLeft(tCos);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsCos()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *mull = new FOfSeVarNode();
				FOfSeVarNode *tSin = new FOfSeVarNode();
				FOfSeVarNode *mOne = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				mOne->SetInt();
				mOne->SetIntVal(-1);
				tSin->SetSin();
				tSin->SetLeft(lCpy);
				mull->SetMull();
				mull->SetLeft(mOne);
				mull->SetRight(tSin);
				res->SetMull();
				res->SetLeft(mull);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsTan()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *div = new FOfSeVarNode();
				FOfSeVarNode *tPow = new FOfSeVarNode();
				FOfSeVarNode *one = new FOfSeVarNode();
				FOfSeVarNode *two = new FOfSeVarNode();
				FOfSeVarNode *tCos = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);
	
				one->SetInt();
				one->SetIntVal(1);
				two->SetInt();
				two->SetIntVal(2);
				tCos->SetCos();
				tCos->SetLeft(lCpy);
				tPow->SetPow();
				tPow->SetLeft(tCos);
				tPow->SetRight(two);
				div->SetDiv();
				div->SetLeft(one);
				div->SetRight(tPow);
				res->SetMull();
				res->SetLeft(div);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsArcSin()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *div = new FOfSeVarNode();
				FOfSeVarNode *one0 = new FOfSeVarNode();
				FOfSeVarNode *tSqrt = new FOfSeVarNode();
				FOfSeVarNode *minus = new FOfSeVarNode();
				FOfSeVarNode *one1 = new FOfSeVarNode();
				FOfSeVarNode *tPow = new FOfSeVarNode();
				FOfSeVarNode *two = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				two->SetInt();
				two->SetIntVal(2);
				tPow->SetPow();
				tPow->SetLeft(lCpy);
				tPow->SetRight(two);
				one0->SetInt();
				one0->SetIntVal(1);
				minus->SetMinus();
				minus->SetLeft(one0);
				minus->SetRight(tPow);
				tSqrt->SetSqrt();
				tSqrt->SetLeft(minus);
				one1->SetInt();
				one1->SetIntVal(1);
				div->SetDiv();
				div->SetLeft(one1);
				div->SetRight(tSqrt);
				res->SetMull();
				res->SetLeft(div);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsArcCos()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *div = new FOfSeVarNode();
				FOfSeVarNode *one0 = new FOfSeVarNode();
				FOfSeVarNode *tSqrt = new FOfSeVarNode();
				FOfSeVarNode *minus = new FOfSeVarNode();
				FOfSeVarNode *one1 = new FOfSeVarNode();
				FOfSeVarNode *tPow = new FOfSeVarNode();
				FOfSeVarNode *two = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				two->SetInt();
				two->SetIntVal(2);
				tPow->SetPow();
				tPow->SetLeft(lCpy);
				tPow->SetRight(two);
				one0->SetInt();
				one0->SetIntVal(1);
				minus->SetMinus();
				minus->SetLeft(one0);
				minus->SetRight(tPow);
				tSqrt->SetSqrt();
				tSqrt->SetLeft(minus);
				one1->SetInt();
				one1->SetIntVal(-1);
				div->SetDiv();
				div->SetLeft(one1);
				div->SetRight(tSqrt);
				res->SetMull();
				res->SetLeft(div);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsArcTan()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *div = new FOfSeVarNode();
				FOfSeVarNode *plus = new FOfSeVarNode();
				FOfSeVarNode *tPow = new FOfSeVarNode();
				FOfSeVarNode *one0 = new FOfSeVarNode();
				FOfSeVarNode *one1 = new FOfSeVarNode();
				FOfSeVarNode *two = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				two->SetInt();
				two->SetIntVal(2);
				tPow->SetPow();
				tPow->SetLeft(lCpy);
				tPow->SetRight(two);
				one0->SetInt();
				one0->SetIntVal(1);
				plus->SetPlus();
				plus->SetLeft(one0);
				plus->SetRight(tPow);
				one1->SetInt();
				one1->SetIntVal(1);
				div->SetDiv();
				div->SetLeft(one1);
				div->SetRight(plus);
				res->SetMull();
				res->SetLeft(div);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsLog()){
				FOfSeVarNode *res = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				res->SetDiv();
				res->SetLeft(lDer);
				res->SetRight(lCpy);

				return res;
			}
			else if (this->IsSqrt()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *mull = new FOfSeVarNode();
				FOfSeVarNode *tSqrt = new FOfSeVarNode();
				FOfSeVarNode *mTwo = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				tSqrt->SetSqrt();
				tSqrt->SetLeft(lCpy);
				mTwo->SetInt();
				mTwo->SetIntVal(-2);
				mull->SetMull();
				mull->SetLeft(mTwo);
				mull->SetRight(tSqrt);
				res->SetDiv();
				res->SetLeft(lDer);
				res->SetRight(mull);

				return res;
			}
			else if (this->IsExp()){
				FOfSeVarNode *res = new FOfSeVarNode();
				FOfSeVarNode *tExp = new FOfSeVarNode();

				FOfSeVarNode *lCpy = left->Cpy();
				FOfSeVarNode *lDer = left->PDer(indVar);

				tExp->SetExp();
				tExp->SetLeft(lCpy);
				res->SetMull();
				res->SetLeft(tExp);
				res->SetRight(lDer);

				return res;
			}
			else if (this->IsPow()){
				if (right->IsInt()){
					FOfSeVarNode *res = new FOfSeVarNode();
					FOfSeVarNode *t0 = new FOfSeVarNode();
					FOfSeVarNode *t1 = new FOfSeVarNode();
					FOfSeVarNode *mull = new FOfSeVarNode();
					FOfSeVarNode *tPow = new FOfSeVarNode();

					FOfSeVarNode *lCpy = left->Cpy();
					FOfSeVarNode *lDer = left->PDer(indVar);

					t1->SetInt();
					t1->SetIntVal(right->GetIntVal() - 1);
					tPow->SetPow();
					tPow->SetLeft(lCpy);
					tPow->SetRight(t1);
					mull->SetMull();
					mull->SetLeft(tPow);
					mull->SetRight(lDer);
					t0->SetInt();
					t0->SetIntVal(right->GetIntVal());
					res->SetMull();
					res->SetLeft(t0);
					res->SetRight(mull);

					return res;
				}
				else if (right->IsDouble()){
					FOfSeVarNode *res = new FOfSeVarNode();
					FOfSeVarNode *t0 = new FOfSeVarNode();
					FOfSeVarNode *t1 = new FOfSeVarNode();
					FOfSeVarNode *mull = new FOfSeVarNode();
					FOfSeVarNode *tPow = new FOfSeVarNode();

					FOfSeVarNode *lCpy = left->Cpy();
					FOfSeVarNode *lDer = left->PDer(indVar);

					t1->SetDouble();
					t1->SetIntVal(right->GetDoubleVal() - 1.0);
					tPow->SetPow();
					tPow->SetLeft(lCpy);
					tPow->SetRight(t1);
					mull->SetMull();
					mull->SetLeft(tPow);
					mull->SetRight(lDer);
					t0->SetDouble();
					t0->SetDoubleVal(right->GetDoubleVal());
					res->SetMull();
					res->SetLeft(t0);
					res->SetRight(mull);

					return res;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					FOfSeVarNode *plus = new FOfSeVarNode();
					FOfSeVarNode *mull1 = new FOfSeVarNode();
					FOfSeVarNode *tLog = new FOfSeVarNode();
					FOfSeVarNode *tDiv = new FOfSeVarNode();
					FOfSeVarNode *mull2 = new FOfSeVarNode();

					FOfSeVarNode *thisCpy = this->Cpy();
					FOfSeVarNode *rDer = right->PDer(indVar);
					FOfSeVarNode *lCpy0 = left->Cpy();
					tLog->SetLog();
					tLog->SetLeft(lCpy0);
					mull1->SetMull();
					mull1->SetLeft(rDer);
					mull1->SetRight(tLog);
					FOfSeVarNode *rCpy = right->Cpy();
					FOfSeVarNode *lDer = left->PDer(indVar);
					mull2->SetMull();
					mull2->SetLeft(rCpy);
					mull2->SetRight(lDer);
					FOfSeVarNode *lCpy1 = left->Cpy();
					tDiv->SetDiv();
					tDiv->SetLeft(mull2);
					tDiv->SetRight(lCpy1);
					plus->SetPlus();
					plus->SetLeft(mull1);
					plus->SetRight(tDiv);
					res->SetMull();
					res->SetLeft(thisCpy);
					res->SetRight(plus);

					return res;
				}

					
			}
			return NULL;
		}
		void GpuCodeGen(FILE *fp){
			if (this->IsPlus()){
				fprintf(fp,"(");
				left->GpuCodeGen(fp);
				fprintf(fp,"+");
				right->GpuCodeGen(fp);
				fprintf(fp,")");
			}
			else if(this->IsMinus()){
				fprintf(fp,"(");
				left->GpuCodeGen(fp);
				fprintf(fp,"-");
				right->GpuCodeGen(fp);
				fprintf(fp,")");
			}
			else if (this->IsMull()){
				fprintf(fp,"(");
				left->GpuCodeGen(fp);
				fprintf(fp,"*");
				right->GpuCodeGen(fp);
				fprintf(fp,")");
			}
			else if (this->IsDiv()){
				fprintf(fp,"(");
				left->GpuCodeGen(fp);
				fprintf(fp,"/");
				right->GpuCodeGen(fp);
				fprintf(fp,")");
			}
			else if (this->IsInt()){
				fprintf(fp,"%d",intVal);
			}
			else if (this->IsDouble()){
				fprintf(fp,"%f",doubleVal);
			}
			else if (this->IsVar()){
				fprintf(fp,"(i*srcColumns + %d)",varNum);
			}
			else if (this->IsSin()){
				fprintf(fp,"(sin(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsCos()){
				fprintf(fp,"(cos(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsTan()){
				fprintf(fp,"(tan(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsArcSin()){
				fprintf(fp,"(arcsin(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsArcCos()){
				fprintf(fp,"(arccos(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsArcTan()){
				fprintf(fp,"(arctan(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsLog()){
				fprintf(fp,"(log(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsSqrt()){
				fprintf(fp,"(sqrt(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsExp()){
				fprintf(fp,"(exp(");
				left->GpuCodeGen(fp);
				fprintf(fp,"))");
			}
			else if (this->IsPow()){
				fprintf(fp,"pow(");
				left->GpuCodeGen(fp);
				fprintf(fp,",");
				right->GpuCodeGen(fp);
				fprintf(fp,")");
			}
		}
		bool Equal(FOfSeVarNode *tNode){
			if (this->IsPlus() && tNode->IsPlus()){
				return left->Equal(tNode->GetLeft()) && right->Equal(tNode->GetRight());
			}
			else if (this->IsMinus() && tNode->IsMinus()){
				return left->Equal(tNode->GetLeft()) && right->Equal(tNode->GetRight());
			}
			else if (this->IsMull() && tNode->IsMull()){
				return left->Equal(tNode->GetLeft()) && right->Equal(tNode->GetRight());
			}
			else if (this->IsDiv() && tNode->IsDiv()){
				return left->Equal(tNode->GetLeft()) && right->Equal(tNode->GetRight());
			}
			else if (this->IsInt() && tNode->IsInt()){
				return (intVal == tNode->GetIntVal());
			}
			else if (this->IsDouble() && tNode->IsDouble()){
				return (doubleVal == tNode->GetDoubleVal());
			}
			else if (this->IsVar() && tNode->IsVar()){
				return (varNum == tNode->GetVarNum());
			}
			else if (this->IsSin() && tNode->IsSin()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsCos() && tNode->IsCos()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsTan() && tNode->IsTan()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsArcSin() && tNode->IsArcSin()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsArcCos() && tNode->IsArcCos()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsArcTan() && tNode->IsArcTan()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsLog() && tNode->IsLog()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsSqrt() && tNode->IsSqrt()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsExp() && tNode->IsExp()){
				return left->Equal(tNode->GetLeft());
			}
			else if (this->IsPow() && tNode->IsPow()){
				return left->Equal(tNode->GetLeft()) && right->Equal(tNode->GetRight());
			}
			return false;
		}
		bool IsZero(){
			return ((this->IsInt() && intVal == 0) ||
					(this->IsDouble() && doubleVal == 0));
		}
		bool IsOne(){
			return ((this->IsInt() && intVal == 1) || (this->IsDouble() && doubleVal == 1.0));
		}
		FOfSeVarNode *SimplifyH(){
			if (this->IsPlus()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *r = right->SimplifyH();
				
				if (l->IsZero()){
					delete l;
					return r;
				}
				else if (r->IsZero()){
					delete r;
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetPlus();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			else if (this->IsMinus()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *r = right->SimplifyH();
				
				if (l->IsZero()){
					delete l;
					return r;
				}
				else if (r->IsZero()){
					delete r;
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetMinus();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			else if (this->IsMull()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *r = right->SimplifyH();
				
				if (l->IsZero() || r->IsZero()){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(0);
					delete l;
					delete r;
					return res;
				}
				else if (l->IsOne()){
					delete l;
					return r;
				}
				else if (r->IsOne()){
					delete r;
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetMull();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			else if (this->IsDiv()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *r = right->SimplifyH();
				
				if (l->IsZero()){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(0);
					delete l;
					delete r;
					return res;
				}
				else if (r->IsOne()){
					delete r;
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetDiv();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			else if(this->IsInt()){
				return this->Cpy();
			}
			else if (this->IsDouble()){
				return this->Cpy();
			}
			else if (this->IsVar()){
				return this->Cpy();
			}
			else if (this->IsSin()){
				FOfSeVarNode *l = left->SimplifyH();
				if (l->IsZero()){
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetSin();
					res->SetLeft(l);
					return res;
				}

			}
			else if (this->IsCos()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetCos();
				res->SetLeft(l);
				return res;
			}
			else if (this->IsTan()){
				FOfSeVarNode *l = left->SimplifyH();

				if (l->IsZero()){
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetTan();
					res->SetLeft(l);
					return res;
				}
			}
			else if (this->IsArcSin()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcSin();
				res->SetLeft(l);
			}
			else if (this->IsArcCos()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcCos();
				res->SetLeft(l);
			}
			else if (this->IsArcTan()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *res = new FOfSeVarNode();
				res->SetArcTan();
				res->SetLeft(l);
			}
			else if (this->IsLog()){
				FOfSeVarNode *l = left->SimplifyH();

				if (l->IsOne()){
					delete l;
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(0);
					return res;
				}
				else{
					FOfSeVarNode * res = new FOfSeVarNode();
					res->SetLog();
					res->SetLeft(l);
					return res;
				}
			}
			else if (this->IsSqrt()){
				FOfSeVarNode *l = left->SimplifyH();

				if (l->IsZero() || l->IsOne()){
					return l;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetSqrt();
					res->SetLeft(l);
					return res;
				}
			
			}
			else if (this->IsExp()){
				FOfSeVarNode *l = left->SimplifyH();

				if (l->IsZero()){
					delete l;
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(1);
					return res;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetExp();
					res->SetLeft(l);
					return res;
				}
			}
			else if (this->IsPow()){
				FOfSeVarNode *l = left->SimplifyH();
				FOfSeVarNode *r = right->SimplifyH();

				if (l->IsZero()){
					delete r;
					delete l;
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(0);
					return res;
				}
				else if (r->IsZero()){
					delete l;
					delete r;
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(1);
					return res;
				}
				else{
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetPow();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			return NULL;
		}
};

class FOfSeVarNodeOp{
	LList<FOfSeVarNode> gList;
	ASTGen astGen;
	public:
		FOfSeVarNodeOp(){
			this->InitializeASTGen();
		}
		FOfSeVarNode *ConstructFromASTNodeH(ASTNode *astNode){
			if (strcmp(astNode->GetName(),"S") == 0){
				LListNode<ASTNode> *t = astNode->GetHead();
				ASTNode *expr = t->GetData();
				return this->ConstructFromASTNodeH(expr);
			}
			else if (strcmp(astNode->GetName(),"EXPR") == 0){
				//expr->expr + expr1 | expr - expr1 | expr1
				LListNode<ASTNode> *t0 = astNode->GetHead();
				LListNode<ASTNode> *t1 = t0->GetNext();
				if (t1 == NULL){
					ASTNode *expr1 = t0->GetData();
					return this->ConstructFromASTNodeH(expr1);
				}
				else{
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t0->GetData();
					ASTNode *sign = t1->GetData();
					ASTNode *expr1 = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					FOfSeVarNode *r = this->ConstructFromASTNodeH(expr1);
					if (strcmp(sign->GetName(),"+") == 0){
						FOfSeVarNode *res = new FOfSeVarNode();
						res->SetPlus();
						res->SetLeft(l);
						res->SetRight(r);
						return res;
					}
					else{
						FOfSeVarNode *res = new FOfSeVarNode();
						res->SetMinus();
						res->SetLeft(l);
						res->SetRight(r);
						return res;
					}
				}
			}
			else if (strcmp(astNode->GetName(),"EXPR1") == 0){
				//expr1->expr1 * expr2 | expr1 / expr2 | expr2
				LListNode<ASTNode> *t0 = astNode->GetHead();
				LListNode<ASTNode> *t1 = t0->GetNext();
				if (t1 == NULL){
					ASTNode *expr2 = t0->GetData();
					return this->ConstructFromASTNodeH(expr2);
				}
				else{
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr1 = t0->GetData();
					ASTNode *sign = t1->GetData();
					ASTNode *expr2 = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr1);
					FOfSeVarNode *r = this->ConstructFromASTNodeH(expr2);
					if (strcmp(sign->GetName(),"*") == 0){
						FOfSeVarNode *res = new FOfSeVarNode();
						res->SetMull();
						res->SetLeft(l);
						res->SetRight(r);
						return res;
					}
					else{
						FOfSeVarNode *res = new FOfSeVarNode();
						res->SetDiv();
						res->SetLeft(l);
						res->SetRight(r);
						return res;
					}
				}
			}
			else if (strcmp(astNode->GetName(),"EXPR2") == 0){
				//expr2->expr2 ^ expr3 | expr3
				LListNode<ASTNode> *t0 = astNode->GetHead();
				LListNode<ASTNode> *t1 = t0->GetNext();
				if (t1 == NULL){
					ASTNode *expr3 = t0->GetData();
					return this->ConstructFromASTNodeH(expr3);
				}
				else{
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr2 = t0->GetData();
					ASTNode *expr3 = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr2);
					FOfSeVarNode *r = this->ConstructFromASTNodeH(expr3);
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetPow();
					res->SetLeft(l);
					res->SetRight(r);
					return res;
				}
			}
			else{
				LListNode<ASTNode> *t0 = astNode->GetHead();
				ASTNode *r0 = t0->GetData();
				if (strcmp(r0->GetName(),"INT") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetInt();
					res->SetIntVal(atoi(r0->GetData()));
					return res;
				}
				else if (strcmp(r0->GetName(),"DOUBLE") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetDouble();
					res->SetDoubleVal((double)atof(r0->GetData()));
					return res;
				}
				else if (strcmp(r0->GetName(),"VAR") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetVar();
					res->SetVarNum(atoi(r0->GetData() + 1));
					return res;
				}
				else if (strcmp(r0->GetName(),"(") == 0){
					LListNode<ASTNode> *t1 = t0->GetNext();
					ASTNode *expr = t1->GetData();
					return this->ConstructFromASTNodeH(expr);
				}
				else if (strcmp(r0->GetName(),"SIN") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetSin();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"COS") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetCos();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"TAN") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetTan();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"ARCSIN") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetArcSin();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"ARCCOS") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetArcCos();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"ARCTAN") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetArcTan();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"LOG") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetLog();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"SQRT") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetSqrt();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
				else if (strcmp(r0->GetName(),"EXP") == 0){
					FOfSeVarNode *res = new FOfSeVarNode();
					res->SetExp();
					LListNode<ASTNode> *t1 = t0->GetNext();
					LListNode<ASTNode> *t2 = t1->GetNext();
					ASTNode *expr = t2->GetData();
					FOfSeVarNode *l = this->ConstructFromASTNodeH(expr);
					res->SetLeft(l);
					return res;
				}
			}
			return NULL;
		}
		FOfSeVarNode * ConstructFromASTNode(ASTNode *astNode,bool keep = false){
			FOfSeVarNode *res = this->ConstructFromASTNodeH(astNode);
			gList.Add(res,keep);
			return res;
		}
		void InitializeASTGen(){
			LexerNodeRoot *root = new LexerNodeRoot();
			
			LexerNodeLeaf *plus = new LexerNodeLeaf();
			plus->And("+");
			root->Add(plus);

			LexerNodeLeaf *minus = new LexerNodeLeaf();
			minus->And("-");
			root->Add(minus);

			LexerNodeLeaf *mul = new LexerNodeLeaf();
			mul->And("*");
			root->Add(mul);

			LexerNodeLeaf *div = new LexerNodeLeaf();
			div->And("/");
			root->Add(div);

			LexerNodeLeaf *iDigit =  new LexerNodeLeaf();
			iDigit->Or("0123456789");
			LexerNodeStar *iDigitStar = new LexerNodeStar(iDigit);
			root->Add(iDigitStar);
	
			LexerNodeLeaf *dDigit = new LexerNodeLeaf();
			dDigit->Or("0123456789");
			LexerNodeStar *dDigitStar = new LexerNodeStar(dDigit);
			LexerNodeLeaf *dDot = new LexerNodeLeaf();
			dDot->And(".");
			LexerNodeLeaf *dDigit2 = new LexerNodeLeaf();
			dDigit2->Or("0123456789");
			LexerNodeStar *dDigit2Star = new LexerNodeStar(dDigit2);
			LexerNodeAnd *dDouble = new LexerNodeAnd();
			dDouble->Add(dDigitStar);
			dDouble->Add(dDot);
			dDouble->Add(dDigit2Star);
			root->Add(dDouble);
			
			LexerNodeLeaf *dollar = new LexerNodeLeaf();
			dollar->And("$");
			LexerNodeLeaf *varDigit = new LexerNodeLeaf();
			varDigit->Or("0123456789");
			LexerNodeStar *varDigitStar = new LexerNodeStar(varDigit);
			LexerNodeAnd *var = new LexerNodeAnd();
			var->Add(dollar);
			var->Add(varDigitStar);
			root->Add(var);

			LexerNodeLeaf *rPar = new LexerNodeLeaf();
			rPar->And("(");
			root->Add(rPar);

			LexerNodeLeaf *lPar = new LexerNodeLeaf();
			lPar->And(")");
			root->Add(lPar);

			LexerNodeLeaf *tSin = new LexerNodeLeaf();
			tSin->And("sin");
			root->Add(tSin);

			LexerNodeLeaf *tCos = new LexerNodeLeaf();
			tCos->And("cos");
			root->Add(tCos);

			LexerNodeLeaf *tTan = new LexerNodeLeaf();
			tTan->And("tan");
			root->Add(tTan);

			LexerNodeLeaf *tArcSin = new LexerNodeLeaf();
			tArcSin->And("arcsin");
			root->Add(tArcSin);

			LexerNodeLeaf *tArcCos = new LexerNodeLeaf();
			tArcCos->And("arccos");
			root->Add(tArcCos);

			LexerNodeLeaf *tArcTan = new LexerNodeLeaf();
			tArcTan->And("arctan");
			root->Add(tArcTan);
			
			LexerNodeLeaf *tLog = new LexerNodeLeaf();
			tLog->And("log");
			root->Add(tLog);

			LexerNodeLeaf *tSqrt = new LexerNodeLeaf();
			tSqrt->And("sqrt");
			root->Add(tSqrt);

			LexerNodeLeaf *tExp = new LexerNodeLeaf();
			tExp->And("exp");
			root->Add(tExp);

			LexerNodeLeaf *tPow = new LexerNodeLeaf();
			tPow->And("^");
			root->Add(tPow);

			Lexer *l = new Lexer(root);
			
			//AST Initialization
			astGen.Allocate(24,5);

			SingleProduction *s = new SingleProduction();
			s->Add(1);

			SingleProduction *expr_0 = new SingleProduction();
			expr_0->Add(1);
			expr_0->Add(5);
			expr_0->Add(2);
			SingleProduction *expr_1 = new SingleProduction();
			expr_1->Add(1);
			expr_1->Add(6);
			expr_1->Add(2);
			SingleProduction *expr_2 = new SingleProduction();
			expr_2->Add(2);

			SingleProduction *expr1_0 = new SingleProduction();
			expr1_0->Add(2);
			expr1_0->Add(7);
			expr1_0->Add(3);
			SingleProduction *expr1_1 = new SingleProduction();
			expr1_1->Add(2);
			expr1_1->Add(8);
			expr1_1->Add(3);
			SingleProduction *expr1_2 = new SingleProduction();
			expr1_2->Add(3);

			SingleProduction *expr2_0 = new SingleProduction();
			expr2_0->Add(3);
			expr2_0->Add(23);
			expr2_0->Add(4);
			SingleProduction *expr2_1 = new SingleProduction();
			expr2_1->Add(4);

			SingleProduction *expr3_0 = new SingleProduction();
			expr3_0->Add(9);

			SingleProduction *expr3_1 = new SingleProduction();
			expr3_1->Add(10);
			
			SingleProduction *expr3_2 = new SingleProduction();
			expr3_2->Add(11);

			SingleProduction *expr3_3 = new SingleProduction();
			expr3_3->Add(12);
			expr3_3->Add(1);
			expr3_3->Add(13);

			SingleProduction *expr3_4 = new SingleProduction();
			expr3_4->Add(14);
			expr3_4->Add(12);
			expr3_4->Add(1);
			expr3_4->Add(13);
			
			SingleProduction *expr3_5 = new SingleProduction();
			expr3_5->Add(15);
			expr3_5->Add(12);
			expr3_5->Add(1);
			expr3_5->Add(13);
			
			SingleProduction *expr3_6 = new SingleProduction();
			expr3_6->Add(16);
			expr3_6->Add(12);
			expr3_6->Add(1);
			expr3_6->Add(13);
			
			SingleProduction *expr3_7 = new SingleProduction();
			expr3_7->Add(17);
			expr3_7->Add(12);
			expr3_7->Add(1);
			expr3_7->Add(13);
			
			SingleProduction *expr3_8 = new SingleProduction();
			expr3_8->Add(18);
			expr3_8->Add(12);
			expr3_8->Add(1);
			expr3_8->Add(13);
			
			SingleProduction *expr3_9 = new SingleProduction();
			expr3_9->Add(19);
			expr3_9->Add(12);
			expr3_9->Add(1);
			expr3_9->Add(13);				

			SingleProduction *expr3_10 = new SingleProduction();
			expr3_10->Add(20);
			expr3_10->Add(12);
			expr3_10->Add(1);
			expr3_10->Add(13);

			SingleProduction *expr3_11 = new SingleProduction();
			expr3_11->Add(21);
			expr3_11->Add(12);
			expr3_11->Add(1);
			expr3_11->Add(13);

			SingleProduction *expr3_12 = new SingleProduction();
			expr3_12->Add(22);
			expr3_12->Add(12);
			expr3_12->Add(1);
			expr3_12->Add(13);
			
			astGen.Add(0,s);
			astGen.Add(1,expr_0);
			astGen.Add(1,expr_1);
			astGen.Add(1,expr_2);
			astGen.Add(2,expr1_0);
			astGen.Add(2,expr1_1);
			astGen.Add(2,expr1_2);
			astGen.Add(3,expr2_0);
			astGen.Add(3,expr2_1);
			astGen.Add(4,expr3_0);
			astGen.Add(4,expr3_1);
			astGen.Add(4,expr3_2);
			astGen.Add(4,expr3_3);
			astGen.Add(4,expr3_4);
			astGen.Add(4,expr3_5);
			astGen.Add(4,expr3_6);
			astGen.Add(4,expr3_7);
			astGen.Add(4,expr3_8);
			astGen.Add(4,expr3_9);
			astGen.Add(4,expr3_10);
			astGen.Add(4,expr3_11);
			astGen.Add(4,expr3_12);

			astGen.SetName(0,"S");
			astGen.SetName(1,"EXPR");
			astGen.SetName(2,"EXPR1");
			astGen.SetName(3,"EXPR2");
			astGen.SetName(4,"EXPR3");
			astGen.SetName(5,"+");
			astGen.SetName(6,"-");
			astGen.SetName(7,"*");
			astGen.SetName(8,"/");
			astGen.SetName(9,"INT");
			astGen.SetName(10,"DOUBLE");
			astGen.SetName(11,"VAR");
			astGen.SetName(12,"(");
			astGen.SetName(13,")");
			astGen.SetName(14,"SIN");
			astGen.SetName(15,"COS");
			astGen.SetName(16,"TAN");
			astGen.SetName(17,"ARCSIN");
			astGen.SetName(18,"ARCCOS");
			astGen.SetName(19,"ARCTAN");
			astGen.SetName(20,"LOG");
			astGen.SetName(21,"SQRT");
			astGen.SetName(22,"EXP");
			astGen.SetName(23,"^");

			astGen.SetLexer(l);
			astGen.ComputeFirstSet();
			astGen.ComputeFollowSet();
			astGen.ConstructRStruct();
		}
		FOfSeVarNode *ConstructFromString(const char *str,bool keep = false){
			ASTNode *funExpr = astGen.GetAST(str);
			FOfSeVarNode *res = this->ConstructFromASTNodeH(funExpr);
			delete funExpr;
			gList.Add(res,keep);
			return res;
		}

};

/*
//Function of several variables
class FOfSeVar{
	int indVar; //number of independant variables
	ASTGen astGen;
	ASTNode *funExpr;
	FOfSeVarNode *root;
	public:
		FOfSeVar(){}
		FOfSeVar(int tIndVar){
			this->Allocate(tIndVar);
		}
		void Allocate(int tIndVar){
			indVar = tIndVar;
			funExpr = NULL;
			root = NULL;
			this->InitializeASTGen();
		}
		~FOfSeVar(){
			delete funExpr;
			delete root;
		}
		void SetRoot(FOfSeVarNode *tRoot){
			root = tRoot;
		}
		void InitializeASTGen(){
			LexerNodeRoot *root = new LexerNodeRoot();
			
			LexerNodeLeaf *plus = new LexerNodeLeaf();
			plus->And("+");
			root->Add(plus);

			LexerNodeLeaf *minus = new LexerNodeLeaf();
			minus->And("-");
			root->Add(minus);

			LexerNodeLeaf *mul = new LexerNodeLeaf();
			mul->And("*");
			root->Add(mul);

			LexerNodeLeaf *div = new LexerNodeLeaf();
			div->And("/");
			root->Add(div);

			LexerNodeLeaf *iDigit =  new LexerNodeLeaf();
			iDigit->Or("0123456789");
			LexerNodeStar *iDigitStar = new LexerNodeStar(iDigit);
			root->Add(iDigitStar);
	
			LexerNodeLeaf *dDigit = new LexerNodeLeaf();
			dDigit->Or("0123456789");
			LexerNodeStar *dDigitStar = new LexerNodeStar(dDigit);
			LexerNodeLeaf *dDot = new LexerNodeLeaf();
			dDot->And(".");
			LexerNodeLeaf *dDigit2 = new LexerNodeLeaf();
			dDigit2->Or("0123456789");
			LexerNodeStar *dDigit2Star = new LexerNodeStar(dDigit2);
			LexerNodeAnd *dDouble = new LexerNodeAnd();
			dDouble->Add(dDigitStar);
			dDouble->Add(dDot);
			dDouble->Add(dDigit2Star);
			root->Add(dDouble);
			
			LexerNodeLeaf *dollar = new LexerNodeLeaf();
			dollar->And("$");
			LexerNodeLeaf *varDigit = new LexerNodeLeaf();
			varDigit->Or("0123456789");
			LexerNodeStar *varDigitStar = new LexerNodeStar(varDigit);
			LexerNodeAnd *var = new LexerNodeAnd();
			var->Add(dollar);
			var->Add(varDigitStar);
			root->Add(var);

			LexerNodeLeaf *rPar = new LexerNodeLeaf();
			rPar->And("(");
			root->Add(rPar);

			LexerNodeLeaf *lPar = new LexerNodeLeaf();
			lPar->And(")");
			root->Add(lPar);

			LexerNodeLeaf *tSin = new LexerNodeLeaf();
			tSin->And("sin");
			root->Add(tSin);

			LexerNodeLeaf *tCos = new LexerNodeLeaf();
			tCos->And("cos");
			root->Add(tCos);

			LexerNodeLeaf *tTan = new LexerNodeLeaf();
			tTan->And("tan");
			root->Add(tTan);

			LexerNodeLeaf *tArcSin = new LexerNodeLeaf();
			tArcSin->And("arcsin");
			root->Add(tArcSin);

			LexerNodeLeaf *tArcCos = new LexerNodeLeaf();
			tArcCos->And("arccos");
			root->Add(tArcCos);

			LexerNodeLeaf *tArcTan = new LexerNodeLeaf();
			tArcTan->And("arctan");
			root->Add(tArcTan);
			
			LexerNodeLeaf *tLog = new LexerNodeLeaf();
			tLog->And("log");
			root->Add(tLog);

			LexerNodeLeaf *tSqrt = new LexerNodeLeaf();
			tSqrt->And("sqrt");
			root->Add(tSqrt);

			LexerNodeLeaf *tExp = new LexerNodeLeaf();
			tExp->And("exp");
			root->Add(tExp);

			LexerNodeLeaf *tPow = new LexerNodeLeaf();
			tPow->And("^");
			root->Add(tPow);

			Lexer *l = new Lexer(root);
			
			//AST Initialization
			astGen.Allocate(24,5);

			SingleProduction *s = new SingleProduction();
			s->Add(1);

			SingleProduction *expr_0 = new SingleProduction();
			expr_0->Add(1);
			expr_0->Add(5);
			expr_0->Add(2);
			SingleProduction *expr_1 = new SingleProduction();
			expr_1->Add(1);
			expr_1->Add(6);
			expr_1->Add(2);
			SingleProduction *expr_2 = new SingleProduction();
			expr_2->Add(2);

			SingleProduction *expr1_0 = new SingleProduction();
			expr1_0->Add(2);
			expr1_0->Add(7);
			expr1_0->Add(3);
			SingleProduction *expr1_1 = new SingleProduction();
			expr1_1->Add(2);
			expr1_1->Add(8);
			expr1_1->Add(3);
			SingleProduction *expr1_2 = new SingleProduction();
			expr1_2->Add(3);

			SingleProduction *expr2_0 = new SingleProduction();
			expr2_0->Add(3);
			expr2_0->Add(23);
			expr2_0->Add(4);
			SingleProduction *expr2_1 = new SingleProduction();
			expr2_1->Add(4);

			SingleProduction *expr3_0 = new SingleProduction();
			expr3_0->Add(9);

			SingleProduction *expr3_1 = new SingleProduction();
			expr3_1->Add(10);
			
			SingleProduction *expr3_2 = new SingleProduction();
			expr3_2->Add(11);

			SingleProduction *expr3_3 = new SingleProduction();
			expr3_3->Add(12);
			expr3_3->Add(1);
			expr3_3->Add(13);

			SingleProduction *expr3_4 = new SingleProduction();
			expr3_4->Add(14);
			expr3_4->Add(12);
			expr3_4->Add(1);
			expr3_4->Add(13);
			
			SingleProduction *expr3_5 = new SingleProduction();
			expr3_5->Add(15);
			expr3_5->Add(12);
			expr3_5->Add(1);
			expr3_5->Add(13);
			
			SingleProduction *expr3_6 = new SingleProduction();
			expr3_6->Add(16);
			expr3_6->Add(12);
			expr3_6->Add(1);
			expr3_6->Add(13);
			
			SingleProduction *expr3_7 = new SingleProduction();
			expr3_7->Add(17);
			expr3_7->Add(12);
			expr3_7->Add(1);
			expr3_7->Add(13);
			
			SingleProduction *expr3_8 = new SingleProduction();
			expr3_8->Add(18);
			expr3_8->Add(12);
			expr3_8->Add(1);
			expr3_8->Add(13);
			
			SingleProduction *expr3_9 = new SingleProduction();
			expr3_9->Add(19);
			expr3_9->Add(12);
			expr3_9->Add(1);
			expr3_9->Add(13);				

			SingleProduction *expr3_10 = new SingleProduction();
			expr3_10->Add(20);
			expr3_10->Add(12);
			expr3_10->Add(1);
			expr3_10->Add(13);

			SingleProduction *expr3_11 = new SingleProduction();
			expr3_11->Add(21);
			expr3_11->Add(12);
			expr3_11->Add(1);
			expr3_11->Add(13);

			SingleProduction *expr3_12 = new SingleProduction();
			expr3_12->Add(22);
			expr3_12->Add(12);
			expr3_12->Add(1);
			expr3_12->Add(13);
			
			astGen.Add(0,s);
			astGen.Add(1,expr_0);
			astGen.Add(1,expr_1);
			astGen.Add(1,expr_2);
			astGen.Add(2,expr1_0);
			astGen.Add(2,expr1_1);
			astGen.Add(2,expr1_2);
			astGen.Add(3,expr2_0);
			astGen.Add(3,expr2_1);
			astGen.Add(4,expr3_0);
			astGen.Add(4,expr3_1);
			astGen.Add(4,expr3_2);
			astGen.Add(4,expr3_3);
			astGen.Add(4,expr3_4);
			astGen.Add(4,expr3_5);
			astGen.Add(4,expr3_6);
			astGen.Add(4,expr3_7);
			astGen.Add(4,expr3_8);
			astGen.Add(4,expr3_9);
			astGen.Add(4,expr3_10);
			astGen.Add(4,expr3_11);
			astGen.Add(4,expr3_12);

			astGen.SetName(0,"S");
			astGen.SetName(1,"EXPR");
			astGen.SetName(2,"EXPR1");
			astGen.SetName(3,"EXPR2");
			astGen.SetName(4,"EXPR3");
			astGen.SetName(5,"+");
			astGen.SetName(6,"-");
			astGen.SetName(7,"*");
			astGen.SetName(8,"/");
			astGen.SetName(9,"INT");
			astGen.SetName(10,"DOUBLE");
			astGen.SetName(11,"VAR");
			astGen.SetName(12,"(");
			astGen.SetName(13,")");
			astGen.SetName(14,"SIN");
			astGen.SetName(15,"COS");
			astGen.SetName(16,"TAN");
			astGen.SetName(17,"ARCSIN");
			astGen.SetName(18,"ARCCOS");
			astGen.SetName(19,"ARCTAN");
			astGen.SetName(20,"LOG");
			astGen.SetName(21,"SQRT");
			astGen.SetName(22,"EXP");
			astGen.SetName(23,"^");

			astGen.SetLexer(l);
			astGen.ComputeFirstSet();
			astGen.ComputeFollowSet();
			astGen.ConstructRStruct();
		}
		void SetFun(const char *str){
			funExpr = astGen.GetAST(str);
			FOfSeVarNodeOp op;
			root = op.ConstructFromASTNode(funExpr);
		}
		void PrntExpr(){
			funExpr->Prnt();
		}
		void PrntExprTree(){
			root->Prnt();
		}
		void GpuCodeGenH(FILE *fp,const char *funcName){
			fprintf(fp,"//-This Code is Automaticaly generated-\n");
			fprintf(fp,"template<class T>\n");
			fprintf(fp,"__global__ void %s(T *res,int resLength,int resBach,T *src,int *srcColumns){\n",funcName);
			fprintf(fp,"\tint from = (threadIdx.x + blockIdx.x * blockDim.x) * resBach;\n");
			fprintf(fp,"\tint to = from + resBach;\n\n");
			fprintf(fp,"\tif (to > resLength){\n");
			fprintf(fp,"\t\tto = resLength;\n");
			fprintf(fp,"\t}\n\n");
			fprintf(fp,"\tfor (int i=from; i<to; i++){\n");
			fprintf(fp,"\t\tres[i] = ");
			root->GpuCodeGen(fp);
			fprintf(fp,"\n");
			fprintf(fp,"\t}\n");
			fprintf(fp,"}");
		}
		void GpuCodeGen(const char *fileName,const char *funcName){
			FILE *fp = fopen(fileName,"w");
			if (fp == NULL){
				printf("Error opening the file %s\n",fileName);
			}
			else{
				this->GpuCodeGenH(fp,funcName);
				fclose(fp);
			}
		}
		FOfSeVar *GetPDer(int tIndVar){
			FOfSeVar *res = new FOfSeVar(indVar);
			FOfSeVarNode *der = root->PDer(tIndVar);
			FOfSeVarNode *sDer = der->SimplifyH();
			delete der;
			res->SetRoot(sDer);

			return res;
		}

};
*/
#endif
































