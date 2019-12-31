#include <Servo.h>

#define STOP      0
#define FORWARD   1
#define BACKWARD  2
#define TURNLEFT  3
#define TURNRIGHT 4
#define AUTO  5

int leftMotor1 = 3;
int leftMotor2 = 4;
int rightMotor1 = 8;
int rightMotor2 = 9;

int leftPWM =5;
int rightPWM = 6;

Servo myServo;  //舵机

int inputPin=11;   // 定义超声波信号接收接口
int outputPin=12;  // 定义超声波信号发出接口

int isauto = 0;
void setup() {
  // put your setup code here, to run once:
  //串口初始化
  Serial.begin(9600); 
  //测速引脚初始化
  pinMode(leftMotor1, OUTPUT);
  pinMode(leftMotor2, OUTPUT);
  pinMode(rightMotor1, OUTPUT);
  pinMode(rightMotor2, OUTPUT);
  pinMode(leftPWM, OUTPUT);
  pinMode(rightPWM, OUTPUT);
  //超声波控制引脚初始化
  pinMode(inputPin, INPUT);
  pinMode(outputPin, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  //if(Serial.available()>0)
  //{
    
    //char cmd = Serial.read();//读取蓝牙模块发送到串口的数据
  
    //Serial.print(cmd);
   // motorRun(cmd);
      
  //}  
                                                                                                                                                                                                                    avoidance();
}
void motorRun(int cmd)
{
  analogWrite(leftPWM, 20);  //设置PWM输出，即设置速度
  analogWrite(rightPWM, 20);
  switch(cmd){
    case AUTO:
      Serial.println("AUTO"); //输出状态
      isauto = 1;
      avoidance();
      break;
      
    case FORWARD:
      Serial.println("FORWARD"); //输出状态
      isauto = 0;
      digitalWrite(leftMotor1, HIGH);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, HIGH);
      digitalWrite(rightMotor2, LOW);
      break;
     case BACKWARD:
      Serial.println("BACKWARD"); //输出状态
      isauto = 0;
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, HIGH);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, HIGH);
      break;
     case TURNLEFT:
      Serial.println("TURN  LEFT"); //输出状态
      isauto = 0;
      digitalWrite(leftMotor1, HIGH);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, LOW);
      break;
     case TURNRIGHT:
      Serial.println("TURN  RIGHT"); //输出状态
      isauto = 0;
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, HIGH);
      digitalWrite(rightMotor2, LOW);
      break;
     case STOP:
      Serial.println("STOP"); //输出状态
      isauto = 0;
      digitalWrite(leftMotor1, LOW);
      digitalWrite(leftMotor2, LOW);
      digitalWrite(rightMotor1, LOW);
      digitalWrite(rightMotor2, LOW);
  }
}
void avoidance()
{
  int dis;//距离
  motorRun(FORWARD);
  dis=getDistance(); //中间
  
  if(dis<30)
  {
    motorRun(TURNRIGHT); 
    delay(500); 
  }
  
}
int getDistance()
{
  digitalWrite(outputPin, LOW); // 使发出发出超声波信号接口低电平2μs
  delayMicroseconds(2);
  digitalWrite(outputPin, HIGH); // 使发出发出超声波信号接口高电平10μs，这里是至少10μs
  delayMicroseconds(10);
  digitalWrite(outputPin, LOW); // 保持发出超声波信号接口低电平
  int distance = pulseIn(inputPin, HIGH); // 读出脉冲时间
  distance= distance/58; // 将脉冲时间转化为距离（单位：厘米）
  Serial.println(distance); //输出距离值
 
  if (distance >=30)
  {
    //如果距离小于50厘米返回数据
    return 30;
  }//如果距离小于50厘米小灯熄灭
  else
    return distance;
}

