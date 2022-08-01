#include <Servo.h>
#include "ServoEasing.hpp"

ServoEasing servoX; 
ServoEasing servoY; 

int servoX_speed = 90;
int servoY_speed = 90;

int lastPosX = 84;
int lastPosY = 84;

int tempX=0;
int tempY=0;

struct retVals {        // Declare a local structure 
    int i1, i2;
  };

retVals retXY(String data) {
  int index = data.indexOf(",");
  String x = data.substring(0,index);
  String y = data.substring(index+1, data.length());
  int posX = x.toInt();
  int posY = y.toInt();
  return retVals {posX, posY}; // Return the local structure
}


void setup() {
  // put your setup code here, to run once:
  servoX.attach(9, 84);
  servoY.attach(10, 84);
  
  servoX.easeTo(84);
  servoY.easeTo(84);

  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0){
    String data = Serial.readStringUntil(33);
    auto [deltaX , deltaY] = retXY(data);
    
    if (deltaX != tempX){
      int cmdX = lastPosX + deltaX ;
      lastPosX = cmdX;
      servoX.easeTo(cmdX, servoX_speed);
      tempX = deltaX ;
      }

    
    if (deltaY != tempY){
      int cmdY = lastPosY + deltaY ;
      lastPosY = cmdY;
      servoY.easeTo(cmdY, servoY_speed);
      tempY = deltaY ;
      }
      
    Serial.read();
  }
}
