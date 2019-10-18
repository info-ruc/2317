//蓝牙小车
#include<stdio.h>
#define STOP 0
#define FORWARD 1
#define BACKWARD 2
#define TURNLEFT 3
#define TURNRIGHT 4
#define BACKLEFT 5
#define BACKRIGHT 6

int leftPWM = 5;
int rightPWM = 3;
int leftMotor1 = 4;
int leftMotor2 = 2;
int rightMotor1 = 8;
int rightMotor2 = 7;
//Ardunio端口
int leftEn = 10;
int rightEn = 11;
//左右轮胎pwm输出端口

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  pinMode(leftPWM, OUTPUT);
  pinMode(rightPWM, OUTPUT);
  //将Ardunio端口作为输出端向驱动模块输出
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0)
  {
    char cmd = Serial.read();

    Serial.print(cmd);
    Run(cmd);
  }
}
void Run(int cmd)
{
  analogWrite(leftPWM, 250);
  analogWrite(rightPWM, 200);
  //配平两侧轮胎速度，保证正常直行和后退
  switch (cmd)
  {
    case STOP:
      Serial.println("STOP");
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, LOW);
      break;
    case FORWARD:
      Serial.println("FORWARD");
      digitalWrite(leftMotor1, HIGH);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, HIGH);
      digitalWrite(rightMotor2, LOW);
      break;
    //直行
    case BACKWARD:
      Serial.println("BACKWARD");
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, HIGH);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, HIGH);
      break;
    //倒退
    case TURNLEFT:
      Serial.println("TURNLEFT");
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, HIGH);
      digitalWrite(rightMotor2, LOW);
      break;
    //左转
    case TURNRIGHT:
      Serial.println("TURNRIGHT");
      digitalWrite(leftMotor1, HIGH);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, LOW);
      break;
    //右转
    case BACKLEFT:
      Serial.println("BACKLEFT");
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, HIGH);
      break;
    //左后转
    case BACKRIGHT:
      Serial.println("BACKRIGHT");
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, HIGH);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, LOW);
      break;
      //右后转
  }
}

