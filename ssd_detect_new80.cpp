// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sys/time.h>
#include <iostream>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
//1/5
using namespace cv;
using namespace std;
string myConvert(float Num)
{
  Num=int(1000*Num)*0.001;
  ostringstream oss;
  oss<<Num;
  string str(oss.str());
  return str;
}
VideoWriter myInitFunction(string video_name,int frame_fps,int frame_width,int frame_height,int isColor)
{
  return VideoWriter(video_name, CV_FOURCC('X', 'V', 'I', 'D'),frame_fps,Size(frame_width,frame_height),isColor);
};
void myDrawFunction(IplImage tmp,int label,float score,CvPoint leftTop,CvPoint rightBottom)
{
  string labelText,scoreText;

  char *coconame[] = {
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign",
"parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
"handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
"surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
"broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
"mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
"vase","scissors","teddy bear","hair drier","toothbrush"};

int num[] = {
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,
42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80};

int rc[] = {255,176,255,65,255,106,255,255,227,56,255,8,237,127,64,0,255,0,192,240};
int gc[] = {255,224,153,105,227,90,215.255,168,94,97,46,145,255,224,255,0,255,192,255};
int bc[] = {0,230,18,225,132,205,0,255,105,15,0,84,33,212,208,0,0,127,192,255};


int r=0,g=0,b=0;
for(int j=0;j<80;j++)
{
	if(label==num[j]){
	labelText = coconame[j];
	scoreText = myConvert(score);
	r=rc[j%20];
	g=gc[j%20];
	b=bc[j%20];
	break;
	}
	else continue;
}

    cvRectangle(&tmp,leftTop,rightBottom,CV_RGB(r,g,b),1.8);
    int myX=int(leftTop.x)/5*5;
    int myY=int(rightBottom.y)/5*5+5;
    CvFont font;
    cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX_SMALL,0.6f,0.6f,0,0.5,CV_AA);
    cvPutText(&tmp,labelText.c_str(),cvPoint(myX,myY+10),&font,CV_RGB(r,g,b));
    cvInitFont(&font,CV_FONT_HERSHEY_COMPLEX_SMALL,0.45f,0.45f,0,0.5,CV_AA);
    cvPutText(&tmp,scoreText.c_str(),cvPoint(myX,myY+20),&font,CV_RGB(r,g,b));
  
};
//Forx

class Detector {
 public:
  Detector(const string& model_file,
           const string& weights_file,
           const string& mean_file,
           const string& mean_value);

  std::vector<vector<float> > Detect(const cv::Mat& img);

 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
    detections.push_back(detection);
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.01,
    "Only store detections with score higher than the threshold.");

int main(int argc, char** argv) {
//czy name
struct timeval t1,t2;
float t_all=0;
gettimeofday(&t1,NULL);

  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Do detection using SSD mode.\n"
        "Usage:\n"
        "    ssd_detect [FLAGS] model_file weights_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "examples/ssd/ssd_detect");
    return 1;
  }

  const string& model_file = argv[1];
  const string& weights_file = argv[2];
  const string& mean_file = FLAGS_mean_file;
  const string& mean_value = FLAGS_mean_value;
  const string& file_type = FLAGS_file_type;
  const string& out_file = FLAGS_out_file;
  const float confidence_threshold = FLAGS_confidence_threshold;

  // Initialize the network.
  Detector detector(model_file, weights_file, mean_file, mean_value);

  // Set the output mode.
  std::streambuf* buf = std::cout.rdbuf();
  std::ofstream outfile;
  if (!out_file.empty()) {
    outfile.open(out_file.c_str());
    if (outfile.good()) {
      buf = outfile.rdbuf();
    }
  }
  std::ostream out(buf);

  // Process image one by one.
  std::ifstream infile(argv[3]);
  std::string file;
  while (infile >> file) {
    if (file_type == "image") {
      cv::Mat img = cv::imread(file, -1);
      CHECK(!img.empty()) << "Unable to decode image " << file;
      std::vector<vector<float> > detections = detector.Detect(img);
	
      /* Print the detection results. */
      for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidence_threshold) {
          out << file << " ";
          out << static_cast<int>(d[1]) << " ";
          out << score << " ";
          out << static_cast<int>(d[3] * img.cols) << " ";
          out << static_cast<int>(d[4] * img.rows) << " ";
          out << static_cast<int>(d[5] * img.cols) << " ";
          out << static_cast<int>(d[6] * img.rows) << std::endl;
        }
      }
    } else if (file_type == "video") {
      cv::VideoCapture cap(file);
      if (!cap.isOpened()) {
        LOG(FATAL) << "Failed to open video: " << file;
      }
      cv::Mat img;
      int frame_count = 0;

      //2/5
      //VideoWriter myInitFunction(string video_name,int frame_fps,int frame_width,int frame_height,int isColor)

      //czy
      VideoWriter writer=myInitFunction("myDetect.avi",29,1280,720,1);
      int frame_begin=1,frame_end=75;
      std::vector<vector<float> > detections;
      CvPoint leftTop,rightBottom;
      leftTop.x=0;
      leftTop.y=0;
      rightBottom.x=0;
      rightBottom.y=0;
      while(frame_count<frame_begin)
      {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
	if (frame_count<frame_begin)
        {
          frame_count++;
          continue;
        }
      }
      while (frame_count<frame_end) {
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << file;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
        ///czy

if(frame_count%1==0)
{
        detections = detector.Detect(img);
}
       
        IplImage tmp=IplImage(img);
        for (int i = 0; i < detections.size(); ++i) {
          const vector<float>& d = detections[i];
          // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
          CHECK_EQ(d.size(), 7);
          const float score = d[2];
          if (score >= confidence_threshold) {
            leftTop.x=static_cast<int>(d[3] * img.cols);
            leftTop.y=static_cast<int>(d[4] * img.rows);
            rightBottom.x=static_cast<int>(d[5] * img.cols);
            rightBottom.y=static_cast<int>(d[6] * img.rows);
            myDrawFunction(tmp,static_cast<int>(d[1]),score,leftTop,rightBottom);
          }
        }
        writer.write(img);
        //czy 1/5
        imshow("detect", img);
        waitKey(1);
        //czy

        ++frame_count;
      }
      if (cap.isOpened()) {
        cap.release();
      }
    } else {
      LOG(FATAL) << "Unknown file_type: " << file_type;
    }
  }
gettimeofday(&t2,NULL);
t_all = (t2.tv_sec-t1.tv_sec)*1000000 +t2.tv_usec-t1.tv_usec;
t_all /= 1000000;

cout<<t_all<<endl;
  return 0;
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
