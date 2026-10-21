#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "WiFiManager.h"

// Disable brownout detector for ESP32-CAM (power issues)
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ========================== Pins & Config ==========================

#define I2S_SPEAKER_BCLK 12 
#define I2S_SPEAKER_LRC  14   
#define I2S_SPEAKER_DIN  13   


// OLED SH1106 display setup
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_SDA 21    
#define OLED_SCL 22    

// Configuration flags
#define USE_AUDIO true     
#define USE_OLED  true     

// LED pins for status indicator
#define STATUS_LED  4      
   

// WiFi credentials - UPDATE THESE
const char* ssid = "Meshkat";
const char* password = "TronicsHub";

const char* websocket_host = "192.168.1.17";  
const int websocket_port = 8080;

// IMPROVED I2S configuration for anti-crackling audio
#define SPEAKER_I2S_NUMBER I2S_NUM_0
#define SAMPLE_RATE 16000
#define DMA_BUF_COUNT 8
#define DMA_BUF_LEN 1024
#define AUDIO_BUFFER_SIZE 65536  // INCREASED from 32768 to 65536

// Audio buffering variables
uint8_t audioBuffer[AUDIO_BUFFER_SIZE];
volatile size_t bufferWriteIndex = 0;
volatile size_t bufferReadIndex = 0;
volatile bool bufferOverflow = false;
volatile bool isReceivingAudio = false;
volatile unsigned long lastAudioReceived = 0;
SemaphoreHandle_t audioMutex = NULL;
TaskHandle_t audioTaskHandle = NULL;

WebSocketsClient webSocket;

// Status variables
unsigned long lastBlinkTime = 0;
bool ledState = false;
int connectionStatus = 0; // 0: Disconnected, 1: WiFi connected, 2: WebSocket connected

// Emotion and animation variables
String currentEmotion = "neutral";
String currentCommand = "";
String serverStatus = "idle";
unsigned long emotionStartTime = 0;
bool isPlayingAudio = false;

// ========================== Emotion Config ==========================
struct EmotionConfig {
    int minAngle;
    int maxAngle;
    int centerAngle;
    int moveSpeed;       // 1-10
    int moveFrequency;   // ms
    bool randomMovement;
    bool bobbingMotion;
};

EmotionConfig emotionConfigs[5] = {
    // NEUTRAL
    {80, 100, 90, 2, 5000, true,  false},
    // HAPPY
    {60, 120, 90, 8,  800,  true, true},
    // SAD
    {85, 95,  90, 1,  3000, false, false},
    // EXCITED
    {45, 135, 90, 10, 300,  true, true},
    // ANGRY
    {70, 110, 90, 6,  500,  true, false}
};

// ========================== Debug ==========================
void debugPrint(String message) {
    Serial.println("[DEBUG] " + message);
    Serial.flush();
}

// ========================== SH1106 Drawing ==========================
#if USE_OLED
#ifndef WHITE
  #define WHITE 1
#endif
#ifndef BLACK
  #define BLACK 0
#endif

Adafruit_SH1106G display = Adafruit_SH1106G(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

void drawEye(Adafruit_SH1106G &d, int cx, int cy, int w, int h, int r) {
  if (w <= 0 || h <= 0) return;
  int x = cx - w / 2;
  int y = cy - h / 2;
  if (w - 2 * r > 0) d.fillRect(x + r, y, w - 2 * r, h, WHITE);
  if (h - 2 * r > 0) d.fillRect(x, y + r, w, h - 2 * r, WHITE);
  if (r > 0) {
    d.fillCircle(x + r, y + r, r, WHITE);
    d.fillCircle(x + w - r - 1, y + r, r, WHITE);
    d.fillCircle(x + r, y + h - r - 1, r, WHITE);
    d.fillCircle(x + w - r - 1, y + h - r - 1, r, WHITE);
  }
}

void drawMouth(Adafruit_SH1106G &d, int cx, int cy, int w, int h, int thick, int curve) {
  if (w <= 0) return;
  if (thick < 1) thick = 1;
  float factor = abs(curve) / 7.0;
  if (factor < 0.5) factor = 0.5;
  for (int i = 0; i <= w; i++) {
    float theta = (float)i / (float)w * 3.1415926f;
    int px = cx - w / 2 + i;
    int py = cy;
    if (curve < 0) {
      py = cy - (int)(sinf(theta) * h * factor);  // Smile
    } else if (curve > 0) {
      py = cy + (int)(sinf(theta) * h * factor);  // Sad
    }
    d.drawLine(px, py - thick / 2, px, py + (thick - 1) / 2, WHITE);
  }
}

void drawFace(
  int eyeWR, int eyeHR, int eyeWL, int eyeHL,
  int eyeCXR, int eyeCXL, int eyeCYR, int eyeCYL,
  int eyeR,
  int mouthW, int mouthH, int mouthThick,
  int mouthCX, int mouthCY,
  int mouthCurve) {
  display.clearDisplay();
  drawEye(display, eyeCXR, eyeCYR, eyeWR, eyeHR, eyeR);
  drawEye(display, eyeCXL, eyeCYL, eyeWL, eyeHL, eyeR);
  drawMouth(display, mouthCX, mouthCY, mouthW, mouthH, mouthThick, mouthCurve);
  display.display();
}

void dynoBlinking() {
  for (int i = 0; i >= -38; i -= 5) {
    drawFace(32,45+i, 32,45+i, 100,28, 30,30, 7, 15,4,5, 64,55, 3);
  }
  for (int i = -38; i <= 0; i += 5) {
    drawFace(32,45+i, 32,45+i, 100,28, 30,30, 7, 15,4,5, 64,55, 3);
  }
}

void smileToSad() {
  for (int i = 8; i >= -8; i -= 5) {
    drawFace(32,45, 32,45, 100,28, 30,30, 7, 15,4,5, 64,55, i);
  }
  for (float i = -15; i <= 15; i += 2) {
    display.fillCircle(115, i - 20, 40, BLACK);
    display.fillCircle(13,  i - 20, 40, BLACK);
    display.display();
  }
}

void oledStatusOverlay() {
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.print("");
  display.print(connectionStatus >= 1 ? "" : "");
  display.print("");
  display.print(connectionStatus >= 2 ? "" : "");
  display.display();
}

void drawNeutralFrame() {
  drawFace(32,45, 32,45, 100,28, 30,30, 7, 15,4,5, 64,55, 3);
  oledStatusOverlay();
}

void drawHappyFrame() {
  drawFace(32,45, 32,45, 100,28, 30,30, 7, 15,4,5, 64,55, -8);
  oledStatusOverlay();
}

void drawSadFrame() {
  drawFace(32,45, 32,45, 100,28, 30,30, 7, 15,4,5, 64,55, 8);
  oledStatusOverlay();
}

void drawExcitedFrame() {
  display.clearDisplay();
  drawMouth(display, 64, 55, 15, 4, 5, 8);
  display.fillTriangle(84, 30, 116, 8.5, 116, 52.5, WHITE);
  display.fillTriangle(92, 30, 116, 8.5, 116, 52.5, BLACK);
  display.fillTriangle(44, 30, 12, 8.5, 12, 52.5, WHITE);
  display.fillTriangle(36, 30, 12, 8.5, 12, 52.5, BLACK);
  display.display();
  oledStatusOverlay();
}

void drawAngryFrame() {
  drawFace(32,45, 32,45, 100,28, 30,30, 7, 15,4,5, 64,55, -3);
  for (float i = -15; i <= 25; i += 2) {
    display.fillTriangle(15, i - 15, 117, i - 15, 64, i + 15, BLACK);
    break;
  }
  display.display();
  oledStatusOverlay();
}

void drawEmotionFace(String emotion) {
  debugPrint("Drawing face for emotion: " + emotion);
  if (!USE_OLED) return;
  if (emotion == "happy")       drawHappyFrame();
  else if (emotion == "sad")    drawSadFrame();
  else if (emotion == "excited")drawExcitedFrame();
  else if (emotion == "angry")  drawAngryFrame();
  else                          drawNeutralFrame();
}
#endif // USE_OLED

// ========================== Audio (I2S) ==========================
#if USE_AUDIO
bool InitI2SSpeaker() {
    debugPrint("Initializing I2S for speaker...");
    audioMutex = xSemaphoreCreateMutex();
    if (audioMutex == NULL) {
        debugPrint("Failed to create audio mutex");
        return false;
    }
    i2s_driver_uninstall(SPEAKER_I2S_NUMBER);
    delay(200);
    i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = DMA_BUF_COUNT,
      .dma_buf_len = DMA_BUF_LEN,
      .use_apll = true,
      .tx_desc_auto_clear = true,
      .fixed_mclk = 0
    };
    esp_err_t err = i2s_driver_install(SPEAKER_I2S_NUMBER, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        debugPrint("Failed to install I2S driver: " + String(err));
        return false;
    }
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SPEAKER_BCLK,
        .ws_io_num = I2S_SPEAKER_LRC,
        .data_out_num = I2S_SPEAKER_DIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };
    err = i2s_set_pin(SPEAKER_I2S_NUMBER, &pin_config);
    if (err != ESP_OK) {
        debugPrint("Failed to set I2S pins: " + String(err));
        return false;
    }
    err = i2s_set_clk(SPEAKER_I2S_NUMBER, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
    if (err != ESP_OK) {
        debugPrint("Failed to set I2S clock: " + String(err));
        return false;
    }
    i2s_zero_dma_buffer(SPEAKER_I2S_NUMBER);
    int16_t silence[DMA_BUF_LEN] = {0};
    size_t bytes_written;
    for (int i = 0; i < DMA_BUF_COUNT; i++) {
        i2s_write(SPEAKER_I2S_NUMBER, silence, sizeof(silence), &bytes_written, pdMS_TO_TICKS(100));
    }
    debugPrint("I2S speaker initialized successfully");
    return true;
}

void bufferAudioData(uint8_t* data, size_t length) {
    if (audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        for (size_t i = 0; i < length; i++) {
            audioBuffer[bufferWriteIndex] = data[i];
            bufferWriteIndex = (bufferWriteIndex + 1) % AUDIO_BUFFER_SIZE;
            if (bufferWriteIndex == bufferReadIndex) {
                bufferOverflow = true;
                debugPrint("Audio buffer overflow!");
                break;
            }
        }
        xSemaphoreGive(audioMutex);
    }
}

void audioPlaybackTask(void* parameter) {
    int16_t outputBuffer[DMA_BUF_LEN];
    size_t bytes_written;
    unsigned long silenceStart = 0;
    bool wasSilent = false;
    
    debugPrint("Audio playback task started on core " + String(xPortGetCoreID()));
    
    while (true) {
        if (isPlayingAudio) {
            // Check if we have data to play
            bool hasData = (bufferWriteIndex != bufferReadIndex);
            
            if (hasData && audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                size_t samplesAvailable = 0;
                size_t samplesToRead = DMA_BUF_LEN;
                
                // Read 16-bit samples directly (2 bytes per sample)
                while (samplesAvailable < samplesToRead && bufferReadIndex != bufferWriteIndex) {
                    size_t nextIndex = (bufferReadIndex + 1) % AUDIO_BUFFER_SIZE;
                    if (nextIndex == bufferWriteIndex) break;
                    
                    uint8_t lowByte = audioBuffer[bufferReadIndex];
                    bufferReadIndex = nextIndex;
                    uint8_t highByte = audioBuffer[bufferReadIndex];
                    bufferReadIndex = (bufferReadIndex + 1) % AUDIO_BUFFER_SIZE;
                    
                    outputBuffer[samplesAvailable] = (int16_t)((highByte << 8) | lowByte);
                    samplesAvailable++;
                }
                
                xSemaphoreGive(audioMutex);
                
                if (samplesAvailable > 0) {
                    wasSilent = false;
                    
                    esp_err_t result = i2s_write(
                        SPEAKER_I2S_NUMBER, 
                        outputBuffer, 
                        samplesAvailable * sizeof(int16_t),
                        &bytes_written, 
                        pdMS_TO_TICKS(100)
                    );
                    
                    if (result != ESP_OK) {
                        debugPrint("I2S write error: " + String(result));
                    }
                }
            } else {
                // Buffer is empty
                if (!wasSilent) {
                    silenceStart = millis();
                    wasSilent = true;
                }
                
                // Check if we should stop playback
                // CRITICAL: Only stop if we're NOT receiving new data AND buffer has been empty for a while
                unsigned long timeSinceLastData = millis() - lastAudioReceived;
                unsigned long silenceDuration = millis() - silenceStart;
                
                // Stop conditions:
                // 1. Not receiving audio AND buffer empty for 2 seconds
                // OR
                // 2. No data received for 3 seconds (stream ended)
                if ((!isReceivingAudio && silenceDuration > 2000) || timeSinceLastData > 3000) {
                    debugPrint("Audio playback complete - stopping");
                    isPlayingAudio = false;
                    wasSilent = false;
                    
                    // Send completion signal
                    webSocket.sendTXT("{\"type\": \"audio_complete\"}");
                } else {
                    // Still waiting for more data - feed silence
                    memset(outputBuffer, 0, sizeof(outputBuffer));
                    i2s_write(SPEAKER_I2S_NUMBER, outputBuffer, sizeof(outputBuffer), &bytes_written, pdMS_TO_TICKS(10));
                }
            }
        } else {
            // Not playing - feed silence to prevent underruns
            wasSilent = false;
            memset(outputBuffer, 0, sizeof(outputBuffer));
            i2s_write(SPEAKER_I2S_NUMBER, outputBuffer, sizeof(outputBuffer), &bytes_written, pdMS_TO_TICKS(10));
        }
        
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

#endif // USE_AUDIO

// ========================== LED ==========================
// void updateStatusLED() {
//     unsigned long currentTime = millis();
//     switch (connectionStatus) {
//         case 0:
//             if (currentTime - lastBlinkTime >= 1000) {
//                 ledState = !ledState;
//                 digitalWrite(STATUS_LED, ledState ? LOW : HIGH);
//                 lastBlinkTime = currentTime;
//             }
//             break;
//         case 1:
//             if (currentTime - lastBlinkTime >= 500) {
//                 ledState = !ledState;
//                 digitalWrite(STATUS_LED, ledState ? LOW : HIGH);
//                 lastBlinkTime = currentTime;
//             }
//             break;
//         case 2:
//             if (serverStatus == "recording") {
//                 digitalWrite(STATUS_LED, LOW);
//             } else if (serverStatus == "listening") {
//                 if (currentTime - lastBlinkTime >= 2000) {
//                     ledState = !ledState;
//                     digitalWrite(STATUS_LED, ledState ? LOW : HIGH);
//                     lastBlinkTime = currentTime;
//                 }
//             } else {
//                 digitalWrite(STATUS_LED, HIGH);
//             }
//             break;
//     }
// }

// ========================== WiFi / WebSocket ==========================
void handleEmotionChange(String newEmotion) {
    debugPrint("Emotion changing from '" + currentEmotion + "' to '" + newEmotion + "'");
    currentEmotion = newEmotion;
    emotionStartTime = millis();
#if USE_OLED
    drawEmotionFace(currentEmotion);
#endif
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            debugPrint("WebSocket Disconnected!");
            connectionStatus = 1;
            isPlayingAudio = false;
            isReceivingAudio = false;
            serverStatus = "idle";
#if USE_AUDIO
            if (audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                bufferWriteIndex = 0;
                bufferReadIndex = 0;
                bufferOverflow = false;
                xSemaphoreGive(audioMutex);
            }
#endif
            break;
        case WStype_CONNECTED:
            debugPrint("WebSocket Connected to: " + String((char*)payload));
            webSocket.sendTXT("{\"type\": \"status\", \"ready\": true}");
            connectionStatus = 2;
            break;
        case WStype_TEXT:
        {
            debugPrint("Received: " + String((char*)payload));
            DynamicJsonDocument doc(1024);
            DeserializationError error = deserializeJson(doc, payload);
            if (error) {
                debugPrint("JSON error: " + String(error.c_str()));
                return;
            }
            if (doc["type"] == "emotion") {
                String newEmotion = doc["value"].as<String>();
                if (newEmotion != currentEmotion) {
                    handleEmotionChange(newEmotion);
                }
            }
            else if (doc["type"] == "command") {
                currentCommand = doc["value"].as<String>();
                debugPrint("Command: " + currentCommand);
            }
            else if (doc["type"] == "status") {
                serverStatus = doc["value"].as<String>();
                debugPrint("Server Status: " + serverStatus);
#if USE_OLED
                drawEmotionFace(currentEmotion);
#endif
            }
            else if (doc["audio"] == true) {
                if (doc["start"] == true) {
                    debugPrint("Audio stream starting...");
                    isReceivingAudio = true;
                    lastAudioReceived = millis();
                    
                    // Only reset if not already playing
                    if (!isPlayingAudio) {
                        isPlayingAudio = true;
#if USE_AUDIO
                        if (audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                            bufferWriteIndex = 0;
                            bufferReadIndex = 0;
                            bufferOverflow = false;
                            xSemaphoreGive(audioMutex);
                        }
#endif
                    }
                } else if (doc["end"] == true) {
                    debugPrint("Audio stream ended - will continue playing buffer");
                    isReceivingAudio = false;
                    lastAudioReceived = millis();
                }
            }
            else if (doc["type"] == "welcome") {
                debugPrint("Received welcome message from server");
            }
        } break;
        case WStype_BIN:
        #if USE_AUDIO
        {
            if (length > 0) {
                // Update last received time
                lastAudioReceived = millis();
                
                // Auto-start playback when binary audio arrives
                if (!isPlayingAudio) {
                    debugPrint("Binary audio detected - auto-starting playback");
                    isPlayingAudio = true;
                    isReceivingAudio = true;
                    if (audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                        bufferWriteIndex = 0;
                        bufferReadIndex = 0;
                        bufferOverflow = false;
                        xSemaphoreGive(audioMutex);
                    }
                }
                
                bufferAudioData(payload, length);
                
                static unsigned long lastBufferStatus = 0;
                if (millis() - lastBufferStatus > 5000) {
                    size_t bufferUsage = (bufferWriteIndex >= bufferReadIndex) ? 
                                      (bufferWriteIndex - bufferReadIndex) : 
                                      (AUDIO_BUFFER_SIZE - bufferReadIndex + bufferWriteIndex);
                    debugPrint("Buffer usage: " + String(bufferUsage) + " / " + String(AUDIO_BUFFER_SIZE));
                    lastBufferStatus = millis();
                }
            }
        }
        #endif
        break;
    }
}

void connectToWiFi() {
    WiFi.mode(WIFI_STA);

    WiFiManager wm;

    if (!wm.autoConnect("AP-NAME", "AP-PASSWORD")) {
        debugPrint("WiFi connection failed!");
        delay(3000);
        ESP.restart();
    }

    debugPrint("WiFi connected! IP: " + WiFi.localIP().toString());
    connectionStatus = 1;
}

void connectToWebSocket() {
    debugPrint("Connecting to WebSocket server...");
    webSocket.begin(websocket_host, websocket_port, "/");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(3000);
    webSocket.enableHeartbeat(15000, 3000, 2);
}

// ========================== Setup / Loop ==========================
void setup() {
    // Disable brownout
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);
    delay(1000);
    setCpuFrequencyMhz(240);
    debugPrint("=== ESP32-CAM Dino Toy Starting ===");
    debugPrint("CPU Frequency: " + String(getCpuFrequencyMhz()) + " MHz");
    pinMode(STATUS_LED, OUTPUT);
    digitalWrite(STATUS_LED, HIGH); // OFF

#if USE_OLED
    // OLED
    debugPrint("Initializing SH1106 display...");
    Wire.begin(OLED_SDA, OLED_SCL);
    if(!display.begin(0x3C, true)) {
        debugPrint("Display initialization failed!");
    } else {
        display.setRotation(2); 
        display.clearDisplay();
        drawEmotionFace("neutral");
        debugPrint("Display initialized successfully");
    }
#endif

#if USE_AUDIO
    // Audio
    debugPrint("Initializing audio system...");
    if (!InitI2SSpeaker()) {
        debugPrint("Audio initialization failed!");
    } else {
        xTaskCreatePinnedToCore(
            audioPlaybackTask,
            "AudioPlayback",
            8192,
            NULL,
            3,
            &audioTaskHandle,
            1
        );
        debugPrint("Audio system initialized successfully");
    }
#endif

    connectToWiFi();
    connectToWebSocket();

    debugPrint("Setup complete! Ready for connections.");
    smileToSad();
}

void loop() {
    webSocket.loop();
    // updateStatusLED();



    if (WiFi.status() != WL_CONNECTED) {
        debugPrint("WiFi connection lost, attempting to reconnect...");
        connectionStatus = 0;
        connectToWiFi();
        if (WiFi.status() == WL_CONNECTED) {
            connectToWebSocket();
        }
    }

    // Blink occasionally, not every loop
    static unsigned long lastBlink = 0;
    if (millis() - lastBlink > 2000) {  // Every 5 seconds
        // if (!isPlayingAudio) {  // Only blink when not playing audio
        dynoBlinking();
        // }
        lastBlink = millis();
    }

    static unsigned long lastStatusReport = 0;
    if (millis() - lastStatusReport > 1000) {
        debugPrint("Status - WiFi: " + String(WiFi.status() == WL_CONNECTED ? "OK" : "DISCONNECTED") + 
                  ", WS: " + String(connectionStatus >= 2 ? "CONNECTED" : "DISCONNECTED") + 
                  ", Emotion: " + currentEmotion + 
                  ", Playing: " + String(isPlayingAudio ? "YES" : "NO") +
                  ", Receiving: " + String(isReceivingAudio ? "YES" : "NO"));
        lastStatusReport = millis();
#if USE_AUDIO
        if (connectionStatus >= 2) {
            size_t bufferUsage = 0;
            if (audioMutex && xSemaphoreTake(audioMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                if (bufferWriteIndex >= bufferReadIndex) bufferUsage = bufferWriteIndex - bufferReadIndex;
                else bufferUsage = AUDIO_BUFFER_SIZE - bufferReadIndex + bufferWriteIndex;
                xSemaphoreGive(audioMutex);
                debugPrint("Buffer: " + String(bufferUsage) + " bytes");
            }
        }
#endif
#if USE_OLED
        drawEmotionFace(currentEmotion);
#endif
    }
    
    delay(5);
}
