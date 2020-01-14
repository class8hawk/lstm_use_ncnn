#include "mat.h"
#include "net.h"
#include"Dir.hpp"
#include "motoproc.h"
#include "mtcnnFeature.h"
#include<iostream>
#include<fstream>
#include <opencv.hpp>


#include <fstream>

int main(int argc, char *argv[])
{
	

	float inputdata[16 * 512];
	std::ifstream outfilel1;
	outfilel1.open("permuted_data.txt", ios::in);


	

	for (int t = 0; t < 16 * 512; t++) {
		//string temps;
		//getline(outfilel1, temps);
		//string ts;
		outfilel1 >> inputdata[t];
		
		//inputdata[t++]= atof(temps.c_str());

	}



	    outfilel1.close();

		
		int indicatorf[16];
		indicatorf[0] = 0;
		for (int i = 1; i < 16; i++)
		{
			indicatorf[i] = 1;
		}

		ncnn::Net eyenet;
		eyenet.load_param("model//platerec.param");
		eyenet.load_model("model//platerec.bin");
		
		ncnn::Mat in(16,sizeof(int),1);
		ncnn::Mat data(512, 16, sizeof(float), 1);
		memcpy(in.data, indicatorf, 16 * sizeof(int));
		memcpy(data.data, inputdata, 16 * 512* sizeof(float));
		ncnn::Extractor ex = eyenet.create_extractor();
		
		ex.set_light_mode(true);
		ex.input("data", data);
		ex.input("indicator", in);
		ncnn::Mat out;

		ex.extract("lstm2", out);

		ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
		std::vector<float> scores;
		scores.resize(out_flatterned.w);

		std::ofstream outfilel2;
		outfilel2.open("lstm2.txt", ios::out);




		


		


		for (int j = 0; j<out_flatterned.w; j++)
		{
			scores[j] = out_flatterned[j];
			outfilel2 << scores[j]<<endl;
		}

		outfilel2.close();

		

		
	return 0;
}
