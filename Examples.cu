#include "Examples.h"

void Examples::GpuArrayBasic(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = new GpuArray<double>(2,3,4);
	GpuArray<double> *b = new GpuArray<double>(2,3,4);
	GpuArray<double> *c = new GpuArray<double>(2,3,1);
	GpuArray<double> *d = new GpuArray<double>(2,1,4);
	GpuArray<double> *e = new GpuArray<double>(2,3,4);
	op.Init(a,1.0);
	op.Init(b,1.0);
	op.LinearInit(c);
	op.LinearInit(d);
	op.LinearInit(e);

	GpuArray<double> *f = op.Add(c,1.0);
	GpuArray<double> *g = op.Add(d,1.0);
	GpuArray<double> *h = op.Add(e,1.0);
	GpuArray<double> *i = new GpuArray<double>(2,4,8);
	op.Init(i,1.0);
	
	std::cout<<"a = \n";
	a->Prnt();
	std::cout<<"b = \n";
	b->Prnt();
	std::cout<<"c = \n";
	c->Prnt();
	std::cout<<"d = \n";
	d->Prnt();
	std::cout<<"e = \n";
	e->Prnt();
	std::cout<<"f = \n";
	f->Prnt();
	std::cout<<"g = \n";
	g->Prnt();
	std::cout<<"h = \n";
	h->Prnt();
	std::cout<<"i = \n";
	i->Prnt();
	
	//Add
	GpuArray<double> *r1 = op.Add(a,b);
	std::cout<<"a + b = \n";
	r1->Prnt();
	//Add Bcast
	GpuArray<double> *r2 = op.Add(a,c);
	std::cout<<"a + c = \n";
	r2->Prnt();
	GpuArray<double> *r3 = op.Add(a,d);
	std::cout<<"a + d = \n";
	r3->Prnt();
	//Add Cnst
	GpuArray<double> *r4 = op.Add(a,2.0);
	std::cout<<"a + 2 = \n";
	r4->Prnt();

	//Sub
	GpuArray<double> *r5 = op.Sub(a,b);
	std::cout<<"a - b = \n";
	r5->Prnt();
	//Sub Bcast
	GpuArray<double> *r6 = op.Sub(a,c);
	std::cout<<"a - c = \n";
	r6->Prnt();
	GpuArray<double> *r7 = op.Sub(a,d);
	std::cout<<"a - d = \n";
	r7->Prnt();
	//Sub Cnst
	GpuArray<double> *r8 = op.Sub(a,1.0);
	std::cout<<"a - 1 = \n";
	r8->Prnt();

	//Dot
	GpuArray<double> *r9 = op.Dot(a,e);
	std::cout<<"a(dot)e = \n";
	r9->Prnt();
	//Dot Bcast
	GpuArray<double> *r10 = op.Dot(a,c);
	std::cout<<"a(dot)c = \n";
	r10->Prnt();
	GpuArray<double> *r11 = op.Dot(a,d);
	std::cout<<"a(dot)d = \n";
	r11->Prnt();
	//Dot Cnst
	GpuArray<double> *r12 = op.Dot(a,3.14);
	std::cout<<"a(dot)3.14 = \n";
	r12->Prnt();

	//Div
	GpuArray<double> *r13 = op.Div(a,h);
	std::cout<<"a/h = \n";
	r13->Prnt();
	//Div Bcast
	GpuArray<double> *r14 = op.Div(a,f);
	std::cout<<"a/f = \n";
	r14->Prnt();
	GpuArray<double> *r15 = op.Div(a,g);
	std::cout<<"a/g = \n";
	r15->Prnt();
	//Div Cnst
	GpuArray<double> *r16 = op.Div(a,2.0);
	std::cout<<"a/2 = \n";
	r16->Prnt();

	//Sigmoid
	GpuArray<double> *r17 = op.Sigmoid(e);
	std::cout<<"Sigmoid(e) = \n";
	r17->Prnt();

	//Log
	GpuArray<double> *r18 = op.Log(h);
	std::cout<<"Log(h) = \n";
	r18->Prnt();

	//Transpose
	GpuArray<double> *r19 = op.Transpose(e);
	std::cout<<"Transpose(e) = \n";
	r19->Prnt();
	
	GpuArray<double> *r20 = op.Mul(a,i);
	std::cout<<"a*i = \n";
	r20->Prnt();

	//Rnd
	GpuArray<double> *r21 = op.Rnd(2,5,6,-1.0,1.0);
	std::cout<<"Rnd = \n";
	r21->Prnt();

	//Add <-
	a->Add(e);
	std::cout<<"a = \n";
	a->Prnt();

	//Add Bcast <-
	std::cout<<"c = \n";
	c->Prnt();
	a->Add(c);
	std::cout<<"a = \n";
	a->Prnt();
	std::cout<<"d = \n";
	d->Prnt();
	a->Add(d);
	std::cout<<"a = \n";
	a->Prnt();

	//Add Cnst <-
	a->Add(3.14);
	std::cout<<"a = \n";
	a->Prnt();

	//Sub <-
	std::cout<<"e = \n";
	e->Prnt();
	std::cout<<"a = a - e\n";
	a->Sub(e);
	std::cout<<"a = \n";
	a->Prnt();

	//Sub Bcast <-
	std::cout<<"c = \n";
	c->Prnt();
	std::cout<<"a = a - c\n";
	a->Sub(c);
	std::cout<<"a = \n";
	a->Prnt();
	std::cout<<"d = \n";
	d->Prnt();
	std::cout<<"a = a - d\n";
	a->Sub(d);
	std::cout<<"a = \n";
	a->Prnt();

	//Sub Cnst <-
	std::cout<<"a = a - 1\n";
	a->Sub(1.0);
	std::cout<<"a = \n";
	a->Prnt();

	//Dot <-
	std::cout<<"a = a(dot)a\n";
	a->Dot(a);
	std::cout<<"a = \n";
	a->Prnt();

	//Dot Bcast <-
	std::cout<<"c = \n";
	c->Prnt();
	std::cout<<"a = a(dot)c\n";
	a->Dot(c);
	std::cout<<"a = \n";
	a->Prnt();
	std::cout<<"d = \n";
	d->Prnt();
	std::cout<<"a = a(dot)d\n";
	a->Dot(d);
	std::cout<<"a = \n";
	a->Prnt();
	
	//Dot Cnst <-
	std::cout<<"a = a(dot)2\n";
	a->Dot(2.0);
	std::cout<<"a = \n";
	a->Prnt();

	//Div <-
	std::cout<<"h = \n";
	h->Prnt();
	std::cout<<"a = a/h\n";
	a->Div(h);
	std::cout<<"a = \n";
	a->Prnt();

	//Div Bcast <-
	std::cout<<"f = \n";
	f->Prnt();
	std::cout<<"a = a/f\n";
	a->Div(f);
	std::cout<<"a = \n";
	a->Prnt();
	std::cout<<"g = \n";
	g->Prnt();
	std::cout<<"a = a/g\n";
	a->Div(g);
	std::cout<<"a = \n";
	a->Prnt();

	//Dot Cnst <-
	std::cout<<"a = a/2\n";
	a->Div(2.0);
	std::cout<<"a = \n";
	a->Prnt();

	//Sigmoid <-
	std::cout<<"h = \n";
	h->Prnt();
	std::cout<<"h = Sigmoid(h)\n";
	h->Sigmoid();
	std::cout<<"h = \n";
	h->Prnt();

	//Log <-
	std::cout<<"h = \n";
	h->Prnt();
	std::cout<<"h = Log(h)\n";
	h->Log();
	std::cout<<"h = \n";
	h->Prnt();

	delete a;
	delete b;
	delete c;
	delete d;
	delete e;
	delete f;
	delete g;
	delete h;
	delete r1;
	delete r2;
	delete r3;
	delete r4;
	delete r5;
	delete r6;
	delete r7;
	delete r8;
	delete r9;
	delete r10;
	delete r11;
	delete r12;
	delete r13;
	delete r14;
	delete r15;
	delete r16;
	delete r17;
	delete r18;
	delete r19;
	delete r20;
	delete r21;
}

void Examples::GpuArrayOpPower(){
	GpuArray<double> *a = new GpuArray<double>(2,3,4);
	GpuArrayOp<double> op(1024,1024);
	op.Init(a,2);
	a->Prnt();
	GpuArray<double> *b = op.Power(a,4);
	b->Prnt();

	delete a;
	delete b;
}

void Examples::GpuArrayOpPw(){
	GpuArray<double> *a = new GpuArray<double>(2,3,4);
	GpuArrayOp<double> op(1024,1024);

	op.Init(a,2);
	a->Prnt();
	GpuArray<double> *b = op.Pw(a,3);
	b->Prnt();

	delete a;
	delete b;
}

void Examples::GpuArrayOpMul3D(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(3,2,3,4,0,1.0);
	GpuArray<double> *b = op.Rnd(3,2,4,5,0,1.0);
	GpuArray<double> *c = op.Mul3D(a,b);
	a->Prnt();
	b->Prnt();
	c->Prnt();

	delete a;
	delete b;
	delete c;
}


void Examples::GpuArrayOpR(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *Rx = op.Rx(input,0);
	GpuArray<double> *Ry = op.Ry(input,1);
	GpuArray<double> *Rz = op.Rz(input,2);

	input->Prnt();
	Rx->Prnt();
	Ry->Prnt();
	Rz->Prnt();

	delete input;
	delete Rx;
	delete Ry;
	delete Rz;
}

void Examples::GpuArrayOpV3D(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *x = op.V3DX(input,0);
	GpuArray<double> *y = op.V3DY(input,1);
	GpuArray<double> *z = op.V3DZ(input,2);

	input->Prnt();
	x->Prnt();
	y->Prnt();
	z->Prnt();

	delete input;
	delete x;
	delete y;
	delete z;
}

void Examples::GpuArrayOpAddToRow(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(2,3,5,0,1.0);
	GpuArray<double> *a1 = op.AddToRow(a,0,0.1);
	GpuArray<double> *a2 = op.AddToRow(a,1,0.1);
	GpuArray<double> *a3 = op.AddToRow(a,2,0.1);

	a->Prnt();
	a1->Prnt();
	a2->Prnt();
	a3->Prnt();

	delete a;
	delete a1;
	delete a2;
	delete a3;
}

void Examples::GpuArrayOpTr3DZ(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(3,5,2,3,0,1.0);
	GpuArray<double> *b = op.Tr3DZ(a);
	a->Prnt();
	b->Prnt();

	delete a;
	delete b;
}

void Examples::GpuArrayOpTpu(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *jac = op.Rnd(3,5,2,3,0,1.0);
	GpuArray<double> *var = op.Rnd(2,5,3,0,1.0);
	GpuArray<double> *tpu = op.Tpu(jac,var);

	jac->Prnt();
	var->Prnt();
	tpu->Prnt();

	delete jac;
	delete var;
	delete tpu;
}

void Examples::GpuArrayOp2DSt0(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *b = op.Rnd(2,3,5,0,1.0);
	GpuArray<double> *c = op._2DSt0(a,b);

	a->Prnt();
	b->Prnt();
	c->Prnt();

	delete a;
	delete b;
	delete c;
}

void Examples::GpuArrayOp2DSt1(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *b = op.Rnd(2,5,4,0,1.0);
	GpuArray<double> *c = op._2DSt1(a,b);

	a->Prnt();
	b->Prnt();
	c->Prnt();

	delete a;
	delete b;
	delete c;
}

void Examples::GpuArrayOp2DFourierSin(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *fourierSin1 = op._2DFourierSin(a,1);
	GpuArray<double> *fourierSin2 = op._2DFourierSin(a,2);
	GpuArray<double> *fourierSin3 = op._2DFourierSin(a,3);

	a->Prnt();
	fourierSin1->Prnt();
	fourierSin2->Prnt();
	fourierSin3->Prnt();

	delete a;
	delete fourierSin1;
	delete fourierSin2;
	delete fourierSin3;
}

void Examples::GpuArrayOp2DFourierCos(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(2,3,4,0,1.0);
	GpuArray<double> *fourierCos1 = op._2DFourierCos(a,1);
	GpuArray<double> *fourierCos2 = op._2DFourierCos(a,2);
	GpuArray<double> *fourierCos3 = op._2DFourierCos(a,3);

	a->Prnt();
	fourierCos1->Prnt();
	fourierCos2->Prnt();
	fourierCos3->Prnt();

	delete a;
	delete fourierCos1;
	delete fourierCos2;
	delete fourierCos3;
}

void Examples::GpuArrayOp3DAlpha(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(3,4,3,6,0,1.0);
	GpuArray<double> *alpha = op._3DAlpha(a);

	a->Prnt();
	alpha->Prnt();

	delete a;
	delete alpha;
}

void Examples::GpuArrayOp3DPlaneL2(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = op.Rnd(3,4,3,6,0,1.0);
	GpuArray<double> *plane = op._3DPlaneL2(a);

	a->Prnt();
	plane->Prnt();

	delete a;
	delete plane;
}

void Examples::GpuArrayPw(){
	GpuArray<double> *a = new GpuArray<double>(2,3,4);
	GpuArrayOp<double> op(1024,1024);

	op.Init(a,2);
	a->Pw(4);
	a->Prnt();

	delete a;
}

void Examples::GpuArraySet3D(){
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *a = new GpuArray<double>(3,4,3,4);
	op.Init(a,0);
	a->Prnt();
	GpuArray<double> *a1 = op.Rnd(2,3,4,0,1.0);
	a1->Prnt();
	a->Set3D(a1,0);
	a->Prnt();
	GpuArray<double> *a2 = op.Rnd(2,3,4,0,1.0);
	a2->Prnt();
	a->Set3D(a2,1);
	a->Prnt();
	GpuArray<double> *a3 = op.Rnd(2,3,4,0,1.0);
	a3->Prnt();
	a->Set3D(a3,2);
	a->Prnt();
	GpuArray<double> *a4 = op.Rnd(2,3,4,0,1.0);
	a4->Prnt();
	a->Set3D(a4,3);
	a->Prnt();

	delete a;
	delete a1;
	delete a2;
	delete a3;
	delete a4;
}

void Examples::DNNSigmoidPrnt(){
	DNNSigmoid a(5,4,7,8,4,5,1024,1024);
	a.PrntWeights();
	a.PrntOffsets();
}

void Examples::DNNSigmoidFProp(){
	DNNSigmoid a(5,4,7,8,4,5,1024,1024);
	
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,4,10,-1.0,1.0);
	
	GpuArray<double> *res = a.FProp(input);
	res->Prnt();

	delete input;
	delete res;
	
}

void Examples::DNNSigmoidLoss(){
	DNNSigmoid a(5,4,7,8,4,2,1024,1024);
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,4,10,-1.0,1.0);
	GpuArray<double> *yHat = new GpuArray<double>(2,2,10);
	yHat->Set(0,0,0);
	yHat->Set(0,1,0);
	yHat->Set(0,2,1);
	yHat->Set(0,3,0);
	yHat->Set(0,4,0);
	yHat->Set(0,5,0);
	yHat->Set(0,6,1);
	yHat->Set(0,7,1);
	yHat->Set(0,8,1);
	yHat->Set(0,9,0);
	yHat->Set(1,0,0);
	yHat->Set(1,1,0);
	yHat->Set(1,2,1);
	yHat->Set(1,3,1);
	yHat->Set(1,4,1);
	yHat->Set(1,5,0);
	yHat->Set(1,6,1);
	yHat->Set(1,7,0);
	yHat->Set(1,8,1);
	yHat->Set(1,9,0);

	std::cout<<a.Loss(input,yHat)<<"\n";

	delete input;
	delete yHat;
}

void Examples::DNNSigmoidTrainBForse(){
	DNNSigmoid a(5,4,7,8,4,2,1024,1024);
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,4,10,-1.0,1.0);
	GpuArray<double> *yHat = new GpuArray<double>(2,2,10);
	yHat->Set(0,0,0);
	yHat->Set(0,1,0);
	yHat->Set(0,2,1);
	yHat->Set(0,3,0);
	yHat->Set(0,4,0);
	yHat->Set(0,5,0);
	yHat->Set(0,6,1);
	yHat->Set(0,7,1);
	yHat->Set(0,8,1);
	yHat->Set(0,9,0);
	yHat->Set(1,0,0);
	yHat->Set(1,1,0);
	yHat->Set(1,2,1);
	yHat->Set(1,3,1);
	yHat->Set(1,4,1);
	yHat->Set(1,5,0);
	yHat->Set(1,6,1);
	yHat->Set(1,7,0);
	yHat->Set(1,8,1);
	yHat->Set(1,9,0);

	a.TrainBForse(input,yHat,100);

	delete input;
	delete yHat;
}

void Examples::DNNSigmoidArrows(){//5013504
	int atributes = 5013504;
	int bachSize = 10;
	int bachNum = 10;
	DNNSigmoid a(3,atributes,10,1,1024,1024);
	GVector <GpuArray <double> > *input = new GVector<GpuArray<double> >(bachNum);
	GVector <GpuArray <double> > *yHat = new GVector<GpuArray<double> >(bachNum);
	CpuArray<double> *cpuInput,*cpuYHat;
	GpuArray<double> *gpuInput,*gpuYHat;
	GpuArrayOp<double> op(1024,1024);
	FILE *fp1,*fp2;

	fp1 = fopen("/home/orestis/JTC_Demo_CameraData/Train/Atributes","r");
	fp2 = fopen("/home/orestis/JTC_Demo_CameraData/Train/Labels","r");
	int val,label;
	for (int i=0; i<bachNum; i++){
		std::cout<<"Bach "<<i<<"\n";
		cpuInput = new CpuArray<double>(2,atributes,bachSize);
		cpuYHat = new CpuArray<double>(2,1,bachSize);
		for (int j=0; j<bachSize; j++){
			std::cout<<"Sample "<<j<<"\n";
			fscanf(fp2,"%d",&label);
			cpuYHat->Set(0,j,(double)label);
			for (int k=0; k<atributes; k++){
				fscanf(fp1,"%d",&val);
				cpuInput->Set(k,j,(double)val/255.0);
			}
		}
		gpuInput = op.Cpy(cpuInput);
		gpuYHat = op.Cpy(cpuYHat);
		delete cpuInput;
		delete cpuYHat;
		input->Set(i,gpuInput);
		yHat->Set(i,gpuYHat);
	}

	std::cout<<"Done Reading...\n";
	std::cout<<"Begining to Train...\n";

	a.TrainBProp(input,yHat,10);
	GpuArray<double> *out;
	
	for (int i=0; i<10; i++){
		out = a.FProp(input->Get(i));
		out->Prnt();
	}

	fclose(fp1);
	fclose(fp2);
	delete input;
	delete yHat;
}

void Examples::VFuncOut(){
	VFunc a(3,4,3,1024,1024);
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,3,10,-1.0,1.0);
	GpuArray<double> *out = a.Out(input);
	out->Prnt();

	delete input;
	delete out;
}

void Examples::VFuncTrain(){
	srand(time(NULL));
	VFunc a(1,6,1,1024,1024);
	GpuArrayOp<double> op(1024,1024);
	GpuArray<double> *input = op.Rnd(2,1,1000,-1.0,1.0);
	GpuArray<double> *f = op.Cos(input);
	a.Train(input,f,50000);
	GpuArray<double> *inputTest = op.Rnd(2,1,5,-1.0,1.0);
	inputTest->Prnt();
	GpuArray<double> *outTest = a.Out(inputTest);
	outTest->Prnt();
	a.PrntWeights();

	delete input;
	delete f;
	delete inputTest;
	delete outTest;
}

void Examples::_LexerNodeLeaf(){
	//
	//And Node
	//
	LexerNodeLeaf leafAnd;
	leafAnd.And("Orestis");
	leafAnd.TransitionMatrixPrnt();

	//
	//Or Node
	//
	LexerNodeLeaf leafOr;
	leafOr.Or("Orestis");
	leafOr.TransitionMatrixPrnt();
}

void Examples::_LexerToken(){
	LexerToken t(0,"Orestis",2,4);
	t.PrntData();
}

void Examples::_Lexer(){
	//Grammar: Orestis|Antwine
	//String: OrestisAntwineAntwineOrestis
	{
		LexerNodeRoot *root = new LexerNodeRoot();
		LexerNodeLeaf *orestis = new LexerNodeLeaf();
		orestis->And("Orestis");
		LexerNodeLeaf *antwine = new LexerNodeLeaf();
		antwine->And("Antwine");
		root->Add(orestis);
		root->Add(antwine);

		Lexer lexer(root);
		lexer.AddString("OrestisAntwineAntwineOrestis");
		LexerToken *t;
		t = lexer.GetNext();
		while(t->GetTokenNum() != -2){
			t->PrntToken();
			delete t;
			t = lexer.GetNext();
		}
	}
	printf("\n\n");
	//Grammar: {123456789}^{0123456789}*|Orestis
	//String: 1234Orestis
	{
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

			LexerNodeLeaf *digit =  new LexerNodeLeaf();
			digit->Or("0123456789");
			LexerNodeStar *digitStar = new LexerNodeStar(digit);
			root->Add(digitStar);

			LexerNodeLeaf *dNZdigit = new LexerNodeLeaf();
			dNZdigit->Or("123456789");
			LexerNodeLeaf *dDigit =  new LexerNodeLeaf();
			dDigit->Or("0123456789");
			LexerNodeStar *dDigitStar = new LexerNodeStar(dDigit);
			LexerNodeAnd *doubleFirstPart = new LexerNodeAnd();
			doubleFirstPart->Add(dNZdigit);
			doubleFirstPart->Add(dDigitStar);
			LexerNodeLeaf *dot = new LexerNodeLeaf();
			dot->And(".");
			LexerNodeLeaf *dDigit2 = new LexerNodeLeaf();
			dDigit2->Or("0123456789");
			LexerNodeStar *dDigitStar2 = new LexerNodeStar(dDigit2);
			LexerNodeAnd *dbl = new LexerNodeAnd();
			dbl->Add(doubleFirstPart);
			dbl->Add(dot);
			dbl->Add(dDigitStar2);
			root->Add(dbl);
	
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
			

		Lexer lexer(root);
		lexer.AddString("sin($2)*3+2*(2+cos($2*$3))");
		LexerToken *t;
		t = lexer.GetNext();
		while(t->GetTokenNum() != -2){
			t->PrntToken();
			delete t;
			t = lexer.GetNext();
		}
	}
}

void Examples::_PrntMatrixConnect(){
	PrntMatrix *prntMatrix = new PrntMatrix(100,100);
	prntMatrix->Init(' ');
	for (int i=0; i<10; i++){
		prntMatrix->Connect(rand()%100,rand()%100,rand()%100,rand()%100);
	}
	prntMatrix->Prnt();
	delete prntMatrix;
}

void Examples::_ASTPrnt(){

	ASTNode *a0 = new ASTNode();
	a0->SetId(0);
	a0->SetName("EXPR");
	
	ASTNode *a1 = new ASTNode();
	a1->SetId(1);
	a1->SetName("EXPR");

	ASTNode *a2 = new ASTNode();
	a2->SetId(2);
	a2->SetTerminal();
	a2->SetName("PLUS");

	ASTNode *a3 = new ASTNode();
	a3->SetId(3);
	a3->SetName("EXPR1");

	ASTNode *a4 = new ASTNode();
	a4->SetId(4);
	a4->SetName("EXPR1");

	ASTNode *a5 = new ASTNode();
	a5->SetId(5);
	a5->SetName("EXPR2");

	ASTNode *a6 = new ASTNode();
	a6->SetId(6);
	a6->SetName("EXPR1");

	ASTNode *a7 = new ASTNode();
	a7->SetId(7);
	a7->SetName("MUL");
	a7->SetTerminal();

	ASTNode *a8 = new ASTNode();
	a8->SetId(8);
	a8->SetName("EXPR2");

	ASTNode *a9 = new ASTNode();
	a9->SetId(9);
	a9->SetTerminal();
	a9->SetName("INT");

	ASTNode *a10 = new ASTNode();
	a10->SetId(10);
	a10->SetName("EXPR2");

	ASTNode *a11 = new ASTNode();
	a11->SetId(11);
	a11->SetName("INT");
	a11->SetTerminal();

	ASTNode *a12 = new ASTNode();
	a12->SetId(12);
	a12->SetName("INT");
	a12->SetTerminal();

	a10->Add(a12);
	a6->Add(a10);
	a8->Add(a11);
	a4->Add(a8);
	a4->Add(a7);
	a4->Add(a6);
	a1->Add(a4);
	a5->Add(a9);
	a3->Add(a5);
	a0->Add(a3);
	a0->Add(a2);
	a0->Add(a1);

	a0->Prnt();

	delete a0;
}

void Examples::_ASTGenFirstSet(){
	SingleProduction *S = new SingleProduction();
	S->Add(1);
	SingleProduction *Expr_0 = new SingleProduction();
	Expr_0->Add(1);
	Expr_0->Add(4);
	Expr_0->Add(2);
	SingleProduction *Expr_1 = new SingleProduction();
	Expr_1->Add(1);
	Expr_1->Add(5);
	Expr_1->Add(2);
	SingleProduction *Expr_2 = new SingleProduction();
	Expr_2->Add(2);
	SingleProduction *Expr1_0 = new SingleProduction();
	Expr1_0->Add(2);
	Expr1_0->Add(6);
	Expr1_0->Add(3);
	SingleProduction *Expr1_1 = new SingleProduction();
	Expr1_1->Add(2);
	Expr1_1->Add(7);
	Expr1_1->Add(3);
	SingleProduction *Expr1_2 = new SingleProduction();
	Expr1_2->Add(3);
	SingleProduction *Expr2_0 = new SingleProduction();
	Expr2_0->Add(8);
	SingleProduction *Expr2_1 = new SingleProduction();
	Expr2_1->Add(9);
	Expr2_1->Add(1);
	Expr2_1->Add(10);

	ASTGen astGen(11,4);
	astGen.Add(0,S);
	astGen.Add(1,Expr_0);
	astGen.Add(1,Expr_1);
	astGen.Add(1,Expr_2);
	astGen.Add(2,Expr1_0);
	astGen.Add(2,Expr1_1);
	astGen.Add(2,Expr1_2);
	astGen.Add(3,Expr2_0);
	astGen.Add(3,Expr2_1);

	astGen.ComputeFirstSet();
	astGen.PrntFirstSet();
}


void Examples::_ASTGenFollowSet(){
	SingleProduction *S = new SingleProduction();
	S->Add(1);
	SingleProduction *Expr_0 = new SingleProduction();
	Expr_0->Add(1);
	Expr_0->Add(4);
	Expr_0->Add(2);
	SingleProduction *Expr_1 = new SingleProduction();
	Expr_1->Add(1);
	Expr_1->Add(5);
	Expr_1->Add(2);
	SingleProduction *Expr_2 = new SingleProduction();
	Expr_2->Add(2);
	SingleProduction *Expr1_0 = new SingleProduction();
	Expr1_0->Add(2);
	Expr1_0->Add(6);
	Expr1_0->Add(3);
	SingleProduction *Expr1_1 = new SingleProduction();
	Expr1_1->Add(2);
	Expr1_1->Add(7);
	Expr1_1->Add(3);
	SingleProduction *Expr1_2 = new SingleProduction();
	Expr1_2->Add(3);
	SingleProduction *Expr2_0 = new SingleProduction();
	Expr2_0->Add(8);
	SingleProduction *Expr2_1 = new SingleProduction();
	Expr2_1->Add(9);
	Expr2_1->Add(1);
	Expr2_1->Add(10);

	ASTGen astGen(11,4);
	astGen.Add(0,S);
	astGen.Add(1,Expr_0);
	astGen.Add(1,Expr_1);
	astGen.Add(1,Expr_2);
	astGen.Add(2,Expr1_0);
	astGen.Add(2,Expr1_1);
	astGen.Add(2,Expr1_2);
	astGen.Add(3,Expr2_0);
	astGen.Add(3,Expr2_1);

	astGen.ComputeFirstSet();
	astGen.ComputeFollowSet();
	astGen.PrntFollowSet();
}

//S->Expr
//Expr->Expr + Expr1 | Expr - Expr1 | Expr1
//Expr1->Expr1 * Expr2 | Expr1 / Expr2 | Expr2
//Expr2->int | (Expr)
//S->0
//Expr->1
//Expr1->2
//Exzpr2->3
//+->4
//-->5
//*->6
///->7
//int->8
//(->9
//)->10
void Examples::_ASTGenStr(){
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
		LexerNodeLeaf *nonZeroDigit = new LexerNodeLeaf();
		nonZeroDigit->Or("123456789");
		LexerNodeLeaf *digit = new LexerNodeLeaf();
		digit->Or("0123456789");
		LexerNodeStar *digitStar = new LexerNodeStar(digit);
		LexerNodeAnd *integer = new LexerNodeAnd();
		integer->Add(nonZeroDigit);
		integer->Add(digitStar);
		root->Add(integer);
		LexerNodeLeaf *rPar = new LexerNodeLeaf();
		rPar->And("(");
		root->Add(rPar);
		LexerNodeLeaf *lPar = new LexerNodeLeaf();
		lPar->And(")");
		root->Add(lPar);

		Lexer *lexer = new Lexer(root);

		SingleProduction *S = new SingleProduction();
		S->Add(1);
		SingleProduction *Expr_0 = new SingleProduction();
		Expr_0->Add(1);
		Expr_0->Add(4);
		Expr_0->Add(2);
		SingleProduction *Expr_1 = new SingleProduction();
		Expr_1->Add(1);
		Expr_1->Add(5);
		Expr_1->Add(2);
		SingleProduction *Expr_2 = new SingleProduction();
		Expr_2->Add(2);
		SingleProduction *Expr1_0 = new SingleProduction();
		Expr1_0->Add(2);
		Expr1_0->Add(6);
		Expr1_0->Add(3);
		SingleProduction *Expr1_1 = new SingleProduction();
		Expr1_1->Add(2);
		Expr1_1->Add(7);
		Expr1_1->Add(3);
		SingleProduction *Expr1_2 = new SingleProduction();
		Expr1_2->Add(3);
		SingleProduction *Expr2_0 = new SingleProduction();
		Expr2_0->Add(8);
		SingleProduction *Expr2_1 = new SingleProduction();
		Expr2_1->Add(9);
		Expr2_1->Add(1);
		Expr2_1->Add(10);

		ASTGen astGen(11,4);
		astGen.Add(0,S);
		astGen.Add(1,Expr_0);
		astGen.Add(1,Expr_1);
		astGen.Add(1,Expr_2);
		astGen.Add(2,Expr1_0);
		astGen.Add(2,Expr1_1);
		astGen.Add(2,Expr1_2);
		astGen.Add(3,Expr2_0);
		astGen.Add(3,Expr2_1);

		astGen.SetName(0,"S");
		astGen.SetName(1,"EXPR");
		astGen.SetName(2,"EXPR1");
		astGen.SetName(3,"EXPR2");
		astGen.SetName(4,"+");
		astGen.SetName(5,"-");
		astGen.SetName(6,"*");
		astGen.SetName(7,"/");
		astGen.SetName(8,"INT");
		astGen.SetName(9,"(");
		astGen.SetName(10,")");

		astGen.SetLexer(lexer);
		astGen.ComputeFirstSet();
		astGen.ComputeFollowSet();
		astGen.ConstructRStruct();

		ASTNode *res = astGen.GetAST("18+2*(3+2)");
		if (res != NULL){
			res->Prnt();
		}
}

void Examples::_FOfSeVarPrntExpr(){
	FOfSeVar f(10);
	f.SetFun("5+2*sin($2*$3)+cos($2*(log($6)))");
	f.PrntExpr();
}

void Examples::_FOfSeVarPrntExprTree(){
	FOfSeVar f(10);
	f.SetFun("5+2*sin($2*$3)+cos($2*(log($6)))");
	f.PrntExprTree();
}

void Examples::_FOfSeVarGpuCodeGen(){
	FOfSeVar f(3);
	f.SetFun("5+2*sin($0*$2)+cos($1*(log($2)))");
	f.GpuCodeGen("GpuCodeGen.cu","MyfirstGpuAutoGeneratedFucntion");
}

void Examples::_FOfSeVarPDer(){
	FOfSeVar f(3);
	f.SetFun("($0)");
	FOfSeVar *fDer = f.GetPDer(0);
	fDer->GpuCodeGen("GpuCodeGen.cu","MyfirstGpuAutoGeneratedFucntion");
	delete fDer;
}






















