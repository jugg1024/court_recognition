#include "plate_recognizor.h"
#include <stdio.h>
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  if (argc < 2) {
  	cout << "param is not enough";
  	return 0;
  }
  PlateRecognizor* pr = new PlateRecognizor(argv[1]);
  pr->recognit(argv[2]);
  return 0;
}


	