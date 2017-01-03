OCRTool : LSMQDFClassification.o MQDFParam.h \
		plate_recognizor.o \
		Bbox.o CharacterFeatureExtraction.o chinese_traverse.o \
		CharacterNormalization.o FeatureStorage.o
	g++ -o OCRTool main.cpp LSMQDFClassification.o MQDFParam.h \
		plate_recognizor.o \
		Bbox.o CharacterFeatureExtraction.o chinese_traverse.o \
		CharacterNormalization.o FeatureStorage.o \
		`pkg-config opencv --cflags --libs` -Wwrite-strings
Bbox.o : Bbox.cpp Bbox.h
	g++ -c Bbox.cpp
CharacterFeatureExtraction.o : CharacterFeatureExtraction.cpp CharacterFeatureExtraction.h 
	g++ -c CharacterFeatureExtraction.cpp
CharacterNormalization.o : CharacterNormalization.cpp CharacterNormalization.h
	g++ -c CharacterNormalization.cpp
LSMQDFClassification.o : LSMQDFClassification.cpp LSMQDFClassification.h
	g++ -c LSMQDFClassification.cpp -std=c++11
chinese_traverse.o : chinese_traverse.cpp chinese_traverse.h Bbox.o CharacterNormalization.o CharacterFeatureExtraction.o
	g++ -c chinese_traverse.cpp
plate_recognizor.o : plate_recognizor.cpp plate_recognizor.h Bbox.o CharacterNormalization.o CharacterFeatureExtraction.o
	g++ -c plate_recognizor.cpp -std=c++11
FeatureStorage.o : FeatureStorage.cpp FeatureStorage.h 
	g++ -c FeatureStorage.cpp -std=c++11
clean :
	rm *.o
	rm OCRTool
