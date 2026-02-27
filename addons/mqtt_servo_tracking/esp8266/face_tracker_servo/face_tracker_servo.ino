#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// Wi-Fi settings
const char* WIFI_SSID = "Yoooo";
const char* WIFI_PASSWORD = "123456789@";

// MQTT settings
const char* MQTT_SERVER = "157.173.101.159";
const uint16_t MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/teamalpha/movement";
const char* MQTT_CLIENT_ID = "teamalpha-face-servo";

// Servo configuration
const uint8_t SERVO_PIN = 14;
const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
const int TRACK_STEP = 2;
const int SEARCH_STEP = 2;
const unsigned long TRACK_INTERVAL_MS = 55;
const unsigned long SEARCH_INTERVAL_MS = 70;
const unsigned long COMMAND_TIMEOUT_MS = 800;

// Set to true if LEFT/RIGHT movement is reversed on your hardware.
const bool REVERSE_SERVO = true;

enum MovementCommand {
  CMD_IDLE,
  CMD_LEFT,
  CMD_RIGHT,
  CMD_CENTER,
  CMD_SEARCH
};

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
Servo panServo;

MovementCommand currentCommand = CMD_IDLE;
int servoAngle = SERVO_CENTER_ANGLE;
int sweepDirection = 1;
unsigned long lastMoveAt = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastCommandAt = 0;

void setServoAngle(int angle) {
  if (angle < SERVO_MIN_ANGLE) {
    angle = SERVO_MIN_ANGLE;
  }
  if (angle > SERVO_MAX_ANGLE) {
    angle = SERVO_MAX_ANGLE;
  }
  servoAngle = angle;
  panServo.write(servoAngle);
}

void applyTrackingStep(int logicalDirection) {
  int direction = REVERSE_SERVO ? -logicalDirection : logicalDirection;
  setServoAngle(servoAngle + (direction * TRACK_STEP));
}

MovementCommand parseCommand(const String& message) {
  if (message == "LEFT") {
    return CMD_LEFT;
  }
  if (message == "RIGHT") {
    return CMD_RIGHT;
  }
  if (message == "CENTER") {
    return CMD_CENTER;
  }
  if (message == "SEARCH") {
    return CMD_SEARCH;
  }
  return CMD_IDLE;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message;
  message.reserve(length);
  for (unsigned int i = 0; i < length; i++) {
    message += static_cast<char>(payload[i]);
  }

  message.trim();
  message.toUpperCase();
  currentCommand = parseCommand(message);
  lastCommandAt = millis();

  Serial.print("[MQTT] topic=");
  Serial.print(topic);
  Serial.print(" cmd=");
  Serial.println(message);

  if (currentCommand == CMD_SEARCH) {
    // Keep current sweep direction when searching.
    return;
  }
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.print("[WiFi] Connecting to ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("[WiFi] Connected. IP: ");
  Serial.println(WiFi.localIP());
}

bool connectMqtt() {
  if (mqttClient.connected()) {
    return true;
  }

  const unsigned long now = millis();
  if (now - lastReconnectAttempt < 2000) {
    return false;
  }
  lastReconnectAttempt = now;

  Serial.print("[MQTT] Connecting to ");
  Serial.print(MQTT_SERVER);
  Serial.print(":");
  Serial.println(MQTT_PORT);

  if (!mqttClient.connect(MQTT_CLIENT_ID)) {
    Serial.print("[MQTT] Connect failed, rc=");
    Serial.println(mqttClient.state());
    return false;
  }

  Serial.println("[MQTT] Connected");
  if (mqttClient.subscribe(MQTT_TOPIC)) {
    Serial.print("[MQTT] Subscribed to ");
    Serial.println(MQTT_TOPIC);
  } else {
    Serial.println("[MQTT] Subscribe failed");
  }
  return true;
}

void handleServo() {
  const unsigned long now = millis();

  // Safety: if commands stop arriving, hold instead of continuing stale movement.
  if ((now - lastCommandAt) > COMMAND_TIMEOUT_MS) {
    currentCommand = CMD_IDLE;
  }

  if (currentCommand == CMD_SEARCH) {
    if (now - lastMoveAt < SEARCH_INTERVAL_MS) {
      return;
    }
    lastMoveAt = now;

    setServoAngle(servoAngle + (sweepDirection * SEARCH_STEP));
    if (servoAngle >= SERVO_MAX_ANGLE) {
      sweepDirection = -1;
    } else if (servoAngle <= SERVO_MIN_ANGLE) {
      sweepDirection = 1;
    }
    return;
  }

  if (now - lastMoveAt < TRACK_INTERVAL_MS) {
    return;
  }
  lastMoveAt = now;

  switch (currentCommand) {
    case CMD_LEFT:
      applyTrackingStep(-1);
      break;
    case CMD_RIGHT:
      applyTrackingStep(1);
      break;
    case CMD_CENTER:
    case CMD_IDLE:
    default:
      // Hold position.
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  panServo.attach(SERVO_PIN);
  setServoAngle(SERVO_CENTER_ANGLE);

  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);

  connectWiFi();
  connectMqtt();
  lastCommandAt = millis();
}

void loop() {
  connectWiFi();
  connectMqtt();
  mqttClient.loop();
  handleServo();
}
