#include <Servo.h>
Servo servoX; 
Servo servoY; 

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
  servoX.attach(9);
  servoY.attach(10);
  
  servoX.write(84);
  servoY.write(84);

  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0){
    String data = Serial.readStringUntil(33);
    auto [deltaX , deltaY] = retXY(data);
    
    if (deltaX != tempX){
      int cmdX = lastPosX + deltaX ;
      lastPosX = cmdX;
      servoX.write(cmdX);
      tempX = deltaX ;
      }

    
    if (deltaY != tempY){
      int cmdY = lastPosY + deltaY ;
      lastPosY = cmdY;
      servoY.write(cmdY);
      tempY = deltaY ;
      }
      
    Serial.read();
  }
}
