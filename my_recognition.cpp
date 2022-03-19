#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("my_recognition_cpp: expected image filename as argument \n");
        printf("example use: ./my_recognition_cpp img.jpg\n");
        return 0;
    }

	// retrieve the image filename from the array of command line args
    const char* imgFilename = argv[1];

    //these vars will store the image data pointers and dimensions
    uchar3* imgPtr = NULL;  //shared CPU/GPU pointer to image
    int imgWidth = 0;       //image width in px
    int imgHeight = 0;      //image height in px

    if(!loadImage(imgFilename, &imgPtr, &imgWidth, &imgHeight))
    {
        printf("Failed to load image '%s' \n" ,imgFilename);
        return 0;
    }

    imageNet* net = imageNet::Create(imageNet::GOOGLENET);

    IF(!net)
    {
        printf("Failed to load image recognition nw\n");
        return 0;
    }

    float confidence = 0.0;
    const int classIndex = net->Classify(imgPtr, imgWidth, imgHeight, &confidence);

    if(classIndex >= 0)
    {
        const char* classDesc = net->GetClassDesc(classIndex);
        printf("index is recognized as '%s' (class #%i) with %.2f%% confidence\n" , classDesc, classIndex, confidence * 100.0f);
    }
    else
        print("Failed to classify image\n");

    delete net;

    return 0;
}