//超声波无舵机小车
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define STOP 0
#define FORWARD 1
#define BACKWARD 2
#define TURNLEFT 3
#define TURNRIGHT 4
#define BACKLEFT 5
#define BACKRIGHT 6

bool start_ultrasonic = false;
bool car_stop = false;
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
int inputPin = 9;
int outputPin = 10;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  pinMode(leftPWM, OUTPUT);
  pinMode(rightPWM, OUTPUT);
  pinMode(inputPin, INPUT);
  pinMode(outputPin, OUTPUT);
  //将Ardunio端口作为输出端向驱动模块输出
}

void loop() {
  // put your main code here, to run repeatedly:
  /*if(Serial.available()>0)
  {
    int cmd = Serial.parseInt();
    
    digitalWrite(outputPin, LOW);
    delayMicroseconds(8);
    digitalWrite(outputPin, HIGH);
    // 维持10毫秒高电平用来产生一个脉冲
    delayMicroseconds(10);
    digitalWrite(outputPin, LOW);
    // 读取脉冲的宽度并换算成距离
    int dist = pulseIn(inputPin, HIGH) / 58.00;
    //超声波测距
    
    Serial.print(dist);
    if(dist < 20 && cmd == FORWARD)
      Run(STOP);
      //如果小车距离前方障碍物仅20cm，则停止小车的继续前进
    else
      Run(cmd);
  }
  */
  //蓝牙+超声波
  /*digitalWrite(outputPin, LOW);
  delayMicroseconds(8);
  digitalWrite(outputPin, HIGH);
  // 维持10毫秒高电平用来产生一个脉冲
  delayMicroseconds(10);
  digitalWrite(outputPin, LOW);
  //读取脉冲的宽度并换算成距离
  int dist = pulseIn(inputPin, HIGH) / 58.00;
  if（dist < 20)
  {
    int a = rand() % 5;
    Run(a);
  }
  else
  {
    Run(STOP);
  }
  delay(2000);*/
  //仅超声波控制
  if(Serial.available()>0)
  {
    //char cmd = Serial.read();
    int cmd = Serial.parseInt();
    Run(cmd);
  }
  //蓝牙控制
}
void Run(int cmd)
{
  analogWrite(leftPWM, 250);
  analogWrite(rightPWM, 200);
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
