#ifndef EXAMPLES_H
#define EXAMPLES_H

#include"GGpuLib.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

class Examples{
	public:
		Examples(){}
		~Examples(){}

		void GpuArrayBasic();

		void GpuArrayOpPower();
		void GpuArrayOpPw();
		void GpuArrayOpMul3D();
		void GpuArrayOpR();
		void GpuArrayOpV3D();
		void GpuArrayOpAddToRow();
		void GpuArrayOpTr3DZ();
		void GpuArrayOpTpu();
		void GpuArrayOp2DSt0();
		void GpuArrayOp2DSt1();
		void GpuArrayOp2DFourierSin();
		void GpuArrayOp2DFourierCos();
		void GpuArrayOp3DAlpha();
		void GpuArrayOp3DPlaneL2();

		void GpuArrayPw();
		void GpuArraySet3D();

		void DNNSigmoidPrnt();
		void DNNSigmoidFProp();
		void DNNSigmoidLoss();
		void DNNSigmoidTrainBForse();
	
		void DNNSigmoidArrows();

		void VFuncOut();
		void VFuncTrain();

		void _LexerNodeLeaf();
		void _LexerToken();
		void _Lexer();

		void _PrntMatrixConnect();

		void _ASTPrnt();

		void _ASTGenFirstSet();
		void _ASTGenFollowSet();
		void _ASTGenStr();

		void _FOfSeVarPrntExpr();
		void _FOfSeVarPrntExprTree();
		void _FOfSeVarGpuCodeGen();
		void _FOfSeVarPDer();
};

#endif
