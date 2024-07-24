#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "stdio.h"
#include "unistd.h"
#include "string.h"
#include <sys/stat.h>
using namespace cv;
using namespace std;

Mat frame,frame1,mask1,mask2,mask3,HSV_frame;
Mat redandblue,harris;
vector<vector<Point>> contours;
vector<vector<Point>> contours1;
vector<vector<Point>> contours2;
vector<Vec4i> hierarchy;
struct stat pic;
float x[10],y[10];
int i=0;
int coordinates[10][2]={0};
int delta_x=0,delta_y=0;
bool is_next=false;
Point2d prev_imagePoint(0,0);
int times_of_change=0;
int length;
Mat cameraMatrix = (Mat_<double>(3, 3) << 3.7006272535101232e+02, 0., 3.1984287642552732e+02, 0.,
       3.6984862245733893e+02, 2.2745772057431958e+02, 0., 0., 1.);
Mat distCoeffs = (Mat_<double>(5, 1) << 1.6150459279761151e-02, -2.3544077627015964e-02,
       -9.1991749702673392e-04, -1.3853766834025158e-03, 0. );
int main()
{
       cv::VideoCapture capture;
       VideoWriter vw;
       capture.open(2); //---------------the index may change-----------------
       int fourcc = vw.fourcc('M','J','P','G');
       capture.set(CAP_PROP_AUTO_EXPOSURE,1);
       capture.set(CAP_PROP_AUTO_WB,0);
       capture.set(CAP_PROP_CONTRAST,10);       
       capture.set(CAP_PROP_FRAME_WIDTH, 1280);//设置摄像头采集图像分辨率
       capture.set(CAP_PROP_FRAME_HEIGHT, 720);       
       capture.set(CAP_PROP_FOURCC,fourcc);
       Rect undistort=Rect(150,100,980,520);
       Rect deblur;
       Mat white(5,5,CV_8UC1,cv::Scalar(255));
       imwrite("./icon.jpg",white);
       waitKey(1000);
       for(;;){      
              std::vector<cv::Point2f> imagePoints;
              std::vector<cv::Point3f> objectPoints;
              cv::Point2f rectPoints[4]={Point(0,0)};
              std::vector<cv::Point2f> perspectivePoints;
              int bot1,bot2;
              stat("./icon.jpg", &pic);
              /*-------------------updated-------------------*/
              capture>>frame1; //取出一帧
              if (frame1.empty()){
                     cout<<"EMPTY IMAGE"<<endl;
                     continue;
              }
              /*-------------------updated-------------------*/

              /*for( int y = 0; y < frame.rows; y++ ) {
                     for( int x = 0; x < frame.cols; x++ ) {
                            for( int c = 0; c < frame.channels(); c++ ) {
                                   frame.at<Vec3b>(y,x)[c] =saturate_cast<uchar>( 1.05*frame.at<Vec3b>(y,x)[c] -20 );
                            }
                     }
              }*/
              imshow("aaaaaaa",frame1);
              waitKey(1);              
              frame=frame1(undistort);           
              cvtColor(frame,HSV_frame,COLOR_BGR2HSV);
              inRange(HSV_frame, Scalar(0,80,100),Scalar(10,255,255),mask1);
              inRange(HSV_frame, Scalar(245,80,100),Scalar(255,255,255),mask2);
              inRange(HSV_frame, Scalar(140,80,100),Scalar(177,255,255),mask3);
              add(mask1, mask2, redandblue); 
              add(mask3, redandblue, redandblue);
              GaussianBlur(redandblue,redandblue,Size(5,5),0,0);
              findContours(redandblue, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());
              if(contours.size()==0){
                     cout<<"no targets"<<endl;
                     if (pic.st_size>400){
                            imwrite("./icon.jpg",white);
                     }
                     continue;
              }
              else{                          
                     for (i = 0; i < contours.size(); i++)//遍历轮廓
                     {
                            int area = contourArea(contours[i]);
                            if (area < 30)drawContours(redandblue, contours, i,Scalar(0,0,0), -1);
                     }
                     Mat kernel = getStructuringElement(MORPH_RECT, Size(35, 35));
                     morphologyEx(redandblue,redandblue,MORPH_CLOSE,kernel,Point(-1,-1),1);                                                
                     findContours(redandblue, contours1, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());//遍历轮廓
                     if(contours1.size()==0){
                            cout<<"NO TARGET"<<endl;
                            if (pic.st_size>400){
                                   imwrite("./icon.jpg",white);
                            }
                            continue;
                     }
                     else{                                                      
                            int area,max=0,maxloc=0;
                            for (i = 0; i < contours1.size(); i++)//遍历轮廓
                            {
                                   area = contourArea(contours1[i]);
                                   if(area>max){
                                          cout<<area<<endl;
                                          max=area;
                                          maxloc=i;
                                   }
                            }
                            if(contourArea(contours1[maxloc])<300){
                                   cout<<"TOO SMALL"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            /*Rect rect = boundingRect(contours1[maxloc]);
                            if(rect.tl().x<=0||rect.tl().y<=0||rect.br().x>=redandblue.cols||rect.br().y>=redandblue.rows){
                                   cout<<"OUT OF BOUNDRY"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }*/
                            drawContours(redandblue, contours1, maxloc,Scalar(150), 3);
                            
                            RotatedRect minrect = minAreaRect(contours1[maxloc]);
                            if(minrect.boundingRect().tl().x<=0||minrect.boundingRect().tl().y<=0||minrect.boundingRect().br().x>=redandblue.cols||minrect.boundingRect().br().y>=redandblue.rows){
                                   cout<<"MINRECT OUT OF BOUNDRY"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            minrect.points(rectPoints);
                            float angle=minrect.angle;
                            if(angle>=89){
                                   cout<<"ANGLE WRONG"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            Mat frame0;
                            Rect rect0(rectPoints[0].x-10,rectPoints[0].y-10,20,20);
                            Rect rect1(rectPoints[1].x-10,rectPoints[1].y-10,20,20);
                            Rect rect2(rectPoints[2].x-10,rectPoints[2].y-10,20,20);
                            Rect rect3(rectPoints[3].x-10,rectPoints[3].y-10,20,20);
                            if(rect0.tl().x<=0||rect0.tl().y<=0||rect0.br().x>=redandblue.cols||rect0.br().y>=redandblue.rows){
                                   cout<<"RECT0 OUT OF BOUNDRY"<<endl;
                                   continue;
                            }
                            if(rect1.tl().x<=0||rect1.tl().y<=0||rect1.br().x>=redandblue.cols||rect1.br().y>=redandblue.rows){
                                   cout<<"RECT1 OUT OF BOUNDRY"<<endl;
                                   continue;
                            }
                            if(rect2.tl().x<=0||rect2.tl().y<=0||rect2.br().x>=redandblue.cols||rect2.br().y>=redandblue.rows){
                                   cout<<"RECT2 OUT OF BOUNDRY"<<endl;
                                   continue;
                            }
                            if(rect3.tl().x<=0||rect3.tl().y<=0||rect3.br().x>=redandblue.cols||rect3.br().y>=redandblue.rows){
                                   cout<<"RECT3 OUT OF BOUNDRY"<<endl;
                                   continue;
                            }
                            Mat redandblue0 = redandblue(rect0);
                            Mat redandblue1 = redandblue(rect1);
                            Mat redandblue2 = redandblue(rect2);
                            Mat redandblue3 = redandblue(rect3);
                            float c0=mean(redandblue0)[0];
                            float c1=mean(redandblue1)[0];
                            float c2=mean(redandblue2)[0];
                            float c3=mean(redandblue3)[0];
                            float c[4]={c0,c1,c2,c3};
                            max=0,maxloc=0;
                            for(int i=0;i<4;i++){
                                   if(c[i]>max){
                                          max=c[i];
                                          maxloc=i;
                                   }
                            }
                            bot1=maxloc,max=0;
                            for(int i=0;i<4;i++){
                                   if(c[i]>max&&i!=bot1){
                                          max=c[i];
                                          maxloc=i;
                                   }
                            }
                            bot2=maxloc;
                            float X=0;
                            float Y=0;
                            for(int i=0;i<4;i++){
                                   if(i!=bot1&&i!=bot2){
                                          X+=rectPoints[i].x;
                                          Y+=rectPoints[i].y;
                                   }
                            }
                            Point Top(X/2,Y/2);
                            Point2f Midbot((rectPoints[bot1].x+rectPoints[bot2].x)/2,(rectPoints[bot1].y+rectPoints[bot2].y)/2);
                            if(Top.x>Midbot.x&&Top.y<Midbot.y){
                                   imagePoints.push_back(Top);
                                   if(rectPoints[bot1].y>rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot2]);
                                          imagePoints.push_back(rectPoints[bot1]);
                                   }
                                   else if(rectPoints[bot1].y<rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot1]);
                                          imagePoints.push_back(rectPoints[bot2]);
                                   }
                                   else{
                                          if (pic.st_size>400){
                                                 imwrite("./icon.jpg",white);
                                          }
                                          continue;
                                   }
                            }
                            else if(Top.x<Midbot.x&&Top.y<Midbot.y){
                                   imagePoints.push_back(Top);
                                   if(rectPoints[bot1].y>rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot1]);
                                          imagePoints.push_back(rectPoints[bot2]);
                                   }
                                   else if(rectPoints[bot1].y<rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot2]);
                                          imagePoints.push_back(rectPoints[bot1]);
                                   }
                                   else{
                                          if (pic.st_size>400){
                                                 imwrite("./icon.jpg",white);
                                          }
                                          continue;
                                   }
                            }
                            else if(Top.x<Midbot.x&&Top.y>Midbot.y){
                                   imagePoints.push_back(Top);
                                   if(rectPoints[bot1].y>rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot1]);
                                          imagePoints.push_back(rectPoints[bot2]);
                                   }
                                   else if(rectPoints[bot1].y<rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot2]);
                                          imagePoints.push_back(rectPoints[bot1]);
                                   }
                                   else{
                                          if (pic.st_size>400){
                                                 imwrite("./icon.jpg",white);
                                          }
                                          continue;
                                   }
                            }
                            else if(Top.x>Midbot.x&&Top.y>Midbot.y){
                                   imagePoints.push_back(Top);
                                   if(rectPoints[bot1].y>rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot2]);
                                          imagePoints.push_back(rectPoints[bot1]);
                                   }
                                   else if(rectPoints[bot1].y<rectPoints[bot2].y){
                                          imagePoints.push_back(rectPoints[bot1]);
                                          imagePoints.push_back(rectPoints[bot2]);
                                   }
                                   else{
                                          if (pic.st_size>400){
                                                 imwrite("./icon.jpg",white);
                                          }
                                          continue;
                                   }
                            }
                            else{
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            //前为判断像素坐标点的顺序
                            Point Middle((Top.x+rectPoints[bot1].x+rectPoints[bot2].x)/3,(Top.y+rectPoints[bot1].y+rectPoints[bot2].y)/3);
                            //中心点坐标
                            imagePoints.push_back(Middle);
                            //世界坐标
                            objectPoints.push_back(cv::Point3f(-707.1, -707.1, 0.));
                            objectPoints.push_back(cv::Point3f(707.1,0.,0.));
                            objectPoints.push_back(cv::Point3f(0.,707.1,0.));
                            objectPoints.push_back(cv::Point3f(0.,0.,0.));
                            //容错
                            if(imagePoints.size()!=4||objectPoints.size()!=4){
                                   cout<<"IMAGEPOINTS or OBJECTPOINTS WRONG"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            //通过像素坐标与世界坐标求解平移向量和旋转矩阵
                            cv::Mat rvec(1,3,cv::DataType<double>::type);
                            cv::Mat tvec(1,3,cv::DataType<double>::type);
                            cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
                            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,SOLVEPNP_ITERATIVE);
                            cv::Rodrigues(rvec,rotationMatrix);
                            //像素坐标中心点视为摄像头在地面的投影点
                            cv::Mat uvPoint = cv::Mat_<double>(3,1);
                            uvPoint.at<double>(0,0)=490;
                            uvPoint.at<double>(1,0)=260;
                            uvPoint.at<double>(2,0)=1.;
                            cv::Mat leftSideMat  = rotationMatrix.inv() * cameraMatrix.inv() * uvPoint;
                            cv::Mat rightSideMat = rotationMatrix.inv() * tvec;
                            double s = (rightSideMat.at<double>(2,0))/leftSideMat.at<double>(2,0);
                            //得到投影点的世界坐标 
                            Mat A=rotationMatrix.inv() * (s * cameraMatrix.inv() * uvPoint - tvec);
                            cout<<"point3:"<<imagePoints[3]<<endl;
                            cout<<"prev point3:"<<prev_imagePoint<<endl;
                            cout<<"projection point:["<<A.at<double>(0,0)<<", "<<A.at<double>(1,0)<<"]"<<endl;
                            if((abs(imagePoints[3].x- prev_imagePoint.x)>300||abs(imagePoints[3].y- prev_imagePoint.y)>180)&&prev_imagePoint.x!=0&&prev_imagePoint.y!=0){
                                   is_next=true; 
                                   times_of_change++;
                                   cout<<"------------------NEXT!----------------------"<<endl;
                            }
                            prev_imagePoint=imagePoints[3];
                            FILE* file = fopen("./coordinates.txt","w");
                            if (file == NULL)
                            {
                                   perror("fopen");
                                   continue;
                            }
                            fprintf(file,"%f\n",A.at<double>(0,0));
                            fprintf(file,"%f",A.at<double>(1,0));
                            if (fclose(file) == EOF)
                            {
                                   //关闭失败
                                   perror("fclose");
                                   return 1;
                            }
                            file = NULL;
                            if(is_next){
                                   FILE* file1 = fopen("./is_next.txt","w");
                                   if (file1 == NULL)
                                   {
                                          perror("fopen");
                                          continue;
                                   }
                                   fprintf(file1,"%d",times_of_change%10);
                                   if (fclose(file1) == EOF)
                                   {
                                          //关闭失败
                                          perror("fclose");
                                          return 1;
                                   }
                                   file1 = NULL;
                                   is_next=false;
                            }
                            cv::Mat rotmat;
                            if(Top.x>Midbot.x&&Top.y<Midbot.y){
                                   rotmat=getRotationMatrix2D(Middle,angle,1);
                            }
                            else if(Top.x<Midbot.x&&Top.y<Midbot.y){
                                   rotmat=getRotationMatrix2D(Middle,270+angle,1);
                            }
                            else if(Top.x<Midbot.x&&Top.y>Midbot.y){
                                   rotmat=getRotationMatrix2D(Middle,180+angle,1);
                            }
                            else if(Top.x>Midbot.x&&Top.y>Midbot.y){
                                   rotmat=getRotationMatrix2D(Middle,90+angle,1);
                            }
                            else{
                                   continue;
                            }
                            warpAffine(frame,frame0,rotmat,Size(980,520));
                            /*//透视变换后想要的像素坐标
                            perspectivePoints.push_back(cv::Point2f(100.,0.));
                            perspectivePoints.push_back(cv::Point2f(0.,300.));
                            perspectivePoints.push_back(cv::Point2f(200.,300.));
                            perspectivePoints.push_back(cv::Point2f(100.,200.));
                            //获取透视矩阵
                            Mat PT = getPerspectiveTransform(imagePoints,perspectivePoints);
                            //透视变换
                            warpPerspective(frame,frame0,PT,Size(200,300));*/
                            if(minrect.size.height<=minrect.size.width){
                                   length=minrect.size.height;
                            }
                            else{
                                   length=minrect.size.width;
                            }
                            Rect num_area(Middle.x-length/4,Middle.y-length/3.8,length/2,length/2);
                            if(num_area.tl().x<=0||num_area.tl().y<=0||num_area.br().x>=frame0.cols||num_area.br().y>=frame0.rows){
                                   cout<<"NUMAREA OUT OF BOUNDRY"<<endl;
                                   if (pic.st_size>400){
                                          imwrite("./icon.jpg",white);
                                   }
                                   continue;
                            }
                            frame0=frame0(num_area);
                            resize(frame0,frame0,Size(128,128));
                            for( int y = 0; y < frame0.rows; y++ ) {
                                   for( int x = 0; x < frame0.cols; x++ ) {
                                          for( int c = 0; c < frame0.channels(); c++ ) {
                                                 frame0.at<Vec3b>(y,x)[c] =saturate_cast<uchar>( 1.1*frame0.at<Vec3b>(y,x)[c]);
                                          }
                                   }
                            }
                            //GaussianBlur(frame0,frame0,Size(7,7),0,0);
                            imwrite("icon.jpg",frame0);
                            /*resize(frame0,frame0,Size(600,600));
                            cvtColor(frame0,frame0,COLOR_BGR2HSV);
                            kernel = getStructuringElement(MORPH_RECT, Size(11,11));
                            inRange(frame0, Scalar(0,0,0),Scalar(255,80,150),mask3);
                            //imshow("first inrange",mask3);
                            //waitKey(1);
                            //GaussianBlur(mask3,mask3,Size(3,3),0,0,4);
                            morphologyEx(mask3,mask3,MORPH_CLOSE,kernel,Point(-1,-1),4);
                            //imshow("close",mask3);
                            //waitKey(1);
                            findContours(mask3, contours2, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());
                            for (i = 0; i < contours2.size(); i++)//遍历轮廓
                            {
                                   int area = contourArea(contours2[i]);
                                   if (area <8000)drawContours(mask3, contours2, i,Scalar(0,0,0), -1);
                            }
                            inRange(mask3, Scalar(60,60,60),Scalar(255,255,255),mask3);
                            //imshow("second inrange",mask3);
                            //waitKey(1);
                            morphologyEx(mask3,mask3,MORPH_DILATE,kernel,Point(-1,-1),3);
                            morphologyEx(mask3,mask3,MORPH_OPEN,kernel,Point(-1,-1),4);
                            //imwrite("eroded.png",mask3);
                            mask3=~mask3;
                            //imshow("numbers",mask3);
                            //waitKey(1);
                            int COL,ROW,blackdots[40]={},COLOR;
                            for(COL=0;COL<40;COL++){
                                   for(ROW=0;ROW<600;ROW++){
                                          COLOR=mask3.at<int>(ROW,COL+280);
                                          if(COLOR<1){
                                                 blackdots[COL]++;
                                          }
                                   }
                            }
                            int min=600,minloc=0;
                            for(int k =0;k<40;k++){
                                   if(blackdots[k]<min){
                                          minloc=k;
                                          min=blackdots[k];
                                   }
                            }*/
                            /*Mat crop1 = frame0(Range(0,260),Range(0,135));
                            Mat crop2 = frame0(Range(0,260),Range(125,260));
                            imwrite("./numbers/aaa.png",crop1);
                            waitKey(5);                  
                            imwrite("./numbers/bbb.png",crop2);
                            waitKey(5);*/
                     }
              }       
       }
}


                            /*rotatedPoints.push_back(Point(rectPoints[0].x,rectPoints[0].y));
                            rotatedPoints.push_back(Point(rectPoints[3].x,rectPoints[3].y));
                            rotatedPoints.push_back(Point(rectPoints[2].x,rectPoints[2].y));
                            rotatedPoints.push_back(Point(rectPoints[1].x,rectPoints[1].y));
                            perspectivePoints.push_back(cv::Point2f(20.,20.));
                            perspectivePoints.push_back(cv::Point2f(20.,280.));
                            perspectivePoints.push_back(cv::Point2f(280.,280.));
                            perspectivePoints.push_back(cv::Point2f(280.,20.));
                            Mat PT = getPerspectiveTransform(rotatedPoints,perspectivePoints);
                            cout<<PT<<endl;
                            warpPerspective(redandblue,redandblue,PT,Size(300,300));
                            morphologyEx(redandblue,redandblue,MORPH_CLOSE,kernel,Point(-1, -1),2);
                            Canny(redandblue,redandblue,100,200,5);
                            imwrite("ptredandblue.jpg",redandblue);
                            Mat frame0;
                            warpPerspective(frame,frame0,PT,Size(300,300));
                            imwrite("ptframe0.jpg",frame0);
                            cornerHarris(redandblue,harris,11,11,0.04);       //寻找Harris角点
                            Mat harrisn;
                            normalize(harris,harrisn,0,255,NORM_MINMAX);
                            convertScaleAbs(harrisn,harrisn);
                            Point2f keyPoint;
                            keyPoint.x=0;
                            keyPoint.y=0;
                            i=0;
                            for(int row=0;row<harrisn.rows;row++){
                                   for(int col=0;col<harrisn.cols;col++){
                                          int R=harrisn.at<uchar>(row,col);
                                          if(R>150){//将角点存入KeyPoint
                                                 keyPoint.y=row;
                                                 keyPoint.x=col;
                                                 i++;
                                                 keyPoints.push_back(keyPoint);
                                          }
                                   }
                            }
                            imwrite("harris.jpg",harrisn);
                            if (i<3){
                                   cout<<"TOO LESS POINTS"<<endl;
                                   continue;
                            }
                            float sum_x=keyPoints[0].x,sum_y=keyPoints[0].y;
                            int k=1,a=0;
                            for(int j=1;j<i;j++){
                                   int dst=abs(keyPoints[j].x-keyPoints[j-1].x)+abs(keyPoints[j].y-keyPoints[j-1].y);
                                   if(dst<18){
                                          sum_x += keyPoints[j].x;
                                          sum_y += keyPoints[j].y;   
                                          k++;
                                   }
                                   if(j==i-1||dst>=30){  
                                          x[a]=(sum_x/k);
                                          y[a]=(sum_y/k);
                                          sum_x=keyPoints[j].x;
                                          sum_y=keyPoints[j].y;
                                          k=1;
                                          a++;
                                   }
                            }
                            
                            if(a!=3){
                                   cout<<a<<endl;
                                   cout<<keyPoints.size()<<endl;
                                   cout<<"Some Y are same"<<endl;
                                   
                                   for(i=0;i<a;i++){
                                   keyPoints[i].x=x[i];
                                   keyPoints[i].y=y[i];
                                   circle(frame0,keyPoints[i],2,Scalar(255,0,0),2);
                                   }
                                   imwrite("error.jpg",frame0);
                                   continue;
                            }
                            for(i=0;i<a;i++){
                                   keyPoints[i].x=x[i];
                                   keyPoints[i].y=y[i];
                                   circle(frame0,keyPoints[i],2,Scalar(255,0,0),1);
                            }
                            imwrite("roi.png",frame0);
                            /*
                            
                            float dist1,dist2,dist3;
                            dist1=(keyPoints[0].x-keyPoints[1].x)*(keyPoints[0].x-keyPoints[1].x)+(keyPoints[0].y-keyPoints[1].y)*(keyPoints[0].y-keyPoints[1].y);
                            dist2=(keyPoints[0].x-keyPoints[2].x)*(keyPoints[0].x-keyPoints[2].x)+(keyPoints[0].y-keyPoints[2].y)*(keyPoints[0].y-keyPoints[2].y);
                            dist3=(keyPoints[2].x-keyPoints[1].x)*(keyPoints[2].x-keyPoints[1].x)+(keyPoints[2].y-keyPoints[1].y)*(keyPoints[2].y-keyPoints[1].y);
                            if(dist1<dist2&&dist1<dist3){
                                   top=2;
                                   bot1=1;
                                   bot2=0;
                            }
                            if(dist2<dist1&&dist2<dist3){
                                   top=1;
                                   bot1=2;
                                   bot2=0;
                            }
                            if(dist3<dist2&&dist3<dist1){
                                   top=0;
                                   bot1=1;
                                   bot2=2;
                            }
                            keyPoints[3].x=(keyPoints[0].x+keyPoints[1].x+keyPoints[2].x)/3;
                            keyPoints[3].y=(keyPoints[0].y+keyPoints[1].y+keyPoints[2].y)/3;
                            Point2i p1,p2,p3,p4;
                            p3.x=keyPoints[bot1].x+keyPoints[bot2].x-keyPoints[3].x;
                            p3.y=keyPoints[bot1].y+keyPoints[bot2].y-keyPoints[3].y;
                            p4.x=2*p3.x-keyPoints[3].x;
                            p4.y=2*p3.y-keyPoints[3].y;
                            p1.x=2*p4.x-keyPoints[3].x;
                            p1.y=2*p4.y-keyPoints[3].y;
                            p2.x=p1.x+10;
                            p2.y=p1.y;
                            float angle[3];
                            for(i=0;i<3;i++){
                                   double a = sqrt(pow(keyPoints[i].x - p2.x, 2) + pow(keyPoints[i].y - p2.y, 2));
                                   double b = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
                                   double c = sqrt(pow(keyPoints[i].x - p1.x, 2) + pow(keyPoints[i].y - p1.y, 2));
                                   angle[i] = acos((b * b + c * c - a * a) / (2 * b * c)) * 180 / CV_PI;
                            }
                            if(keyPoints[top].y<p1.y){
                                   if(angle[bot1]>angle[bot2]){
                                          imagePoints.push_back(keyPoints[top]);
                                          imagePoints.push_back(keyPoints[bot1]);
                                          imagePoints.push_back(keyPoints[bot2]);
                                   }//bigger is anti
                                   if(angle[bot2]>angle[bot1]){
                                          imagePoints.push_back(keyPoints[top]);
                                          imagePoints.push_back(keyPoints[bot2]);
                                          imagePoints.push_back(keyPoints[bot1]);
                                   }
                            }
                            if(keyPoints[top].y>p1.y){
                                   if(angle[bot1]>angle[bot2]){
                                          imagePoints.push_back(keyPoints[top]);
                                          imagePoints.push_back(keyPoints[bot2]);
                                          imagePoints.push_back(keyPoints[bot1]);
                                   }//bigger is anti
                                   if(angle[bot2]>angle[bot1]){
                                          imagePoints.push_back(keyPoints[top]);
                                          imagePoints.push_back(keyPoints[bot1]);
                                          imagePoints.push_back(keyPoints[bot2]);
                                   }//bigger is anti
                            }
                            imagePoints.push_back(keyPoints[3]);
                            if(imagePoints.size()!=4){
                                   cout<<"SIZE OF IMAGEPOINTS IS WRONG"<<endl;
                                   continue;
                            }
                            imwrite("roi.png",frame0);
                            objectPoints.push_back(cv::Point3f(0., 0., 0.));
                            objectPoints.push_back(cv::Point3f(1414.2,707.1,0.));
                            objectPoints.push_back(cv::Point3f(707.1,1414.2,0.));
                            objectPoints.push_back(cv::Point3f(707.1,707.1,0.));
                            Mat cameraMatrix = (Mat_<double>(3, 3) << 3.4559931046298266e+02, 0., 3.0369206210193323e+02, 0.,
       3.4274103201381132e+02, 4.0775425179058118e+02, 0., 0., 1.);
                            Mat distCoeffs = (Mat_<double>(5, 1) << -4.6972863420219725e-03, -4.4530474231755789e-04,
       3.8877908533517259e-04, 5.7240601762703820e-04, 0.  );
                            cv::Mat rvec(1,3,cv::DataType<double>::type);
                            cv::Mat tvec(1,3,cv::DataType<double>::type);
                            cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
                            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,SOLVEPNP_ITERATIVE);
                            Point2i center(frame.cols/2,frame.rows/2);
                            Point2d centerpoint;
                            centerpoint.x=center.x-rect.tl().x;
                            centerpoint.y=center.y-rect.tl().y;
                            cv::Rodrigues(rvec,rotationMatrix);
                            cv::Mat uvPoint = cv::Mat_<double>(3,1);
                            uvPoint.at<double>(0,0)=centerpoint.x;
                            uvPoint.at<double>(1,0)=centerpoint.y;
                            uvPoint.at<double>(2,0)=1.;
                            // u = 363, v = 222, got this point using mouse callback
                            cv::Mat leftSideMat  = rotationMatrix.inv() * cameraMatrix.inv() * uvPoint;
                            cv::Mat rightSideMat = rotationMatrix.inv() * tvec;
                            double s = (rightSideMat.at<double>(2,0))/leftSideMat.at<double>(2,0); 
                            Mat A=rotationMatrix.inv() * (s * cameraMatrix.inv() * uvPoint - tvec);
                            dest.x=A.at<double>(0,0);
                            dest.y=A.at<double>(1,0);
                            cout<<dest<<endl;
                            */
                            /*cv::Mat rotmat;
                            if(Top.x>Midbot.x&&Top.y<Midbot.y){
                                   rotmat=getRotationMatrix2D(minrect.center,angle,1);
                            }
                            else if(Top.x<Midbot.x&&Top.y<Midbot.y){
                                   rotmat=getRotationMatrix2D(minrect.center,270+angle,1);
                            }
                            else if(Top.x<Midbot.x&&Top.y>Midbot.y){
                                   rotmat=getRotationMatrix2D(minrect.center,180+angle,1);
                            }
                            else if(Top.x>Midbot.x&&Top.y>Midbot.y){
                                   rotmat=getRotationMatrix2D(minrect.center,90+angle,1);
                            }
                            else{
                                   continue;
                            }
                            warpAffine(frame,frame0,rotmat,frame.size());*/

                            /*float length;
                            if(minrect.size.height<=minrect.size.width){
                                   length=minrect.size.height;
                            }
                            else{
                                   length=minrect.size.width;
                            }*/
