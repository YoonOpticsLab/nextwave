#include <stdio.h>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <json.hpp>
using json=nlohmann::json;

#include "nextwave_plugin.hpp"

DECL init(char *params)
{
  json jdata = json::parse(params);
  std::string filename=jdata["filename"];

  Mat image;

  image = imread( filename, IMREAD_COLOR );

  if ( !image.data )
  {
    printf("No image data \n");
    return -1;
  }

  namedWindow(filename, WINDOW_AUTOSIZE );
  imshow(filename, image);
  waitKey(0);

  return 0;
}
