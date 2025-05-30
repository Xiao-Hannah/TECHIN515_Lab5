#include <magic_wand_hannah_inferencing.h> 
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#define CONFIDENCE_THRESHOLD 80.0

// MPU sensor
Adafruit_MPU6050 mpu;

// WiFi credentials 
const char* ssid = "UW MPSK";
const char* password = "6-g5WjqCq]";
// Server details 
const char* serverUrl = "http://10.19.110.181:5001/predict";

// Student identifier
const char* studentId = "hx2313";

// LED pins (GPIO)
#define YELLOW_LED D2  // GPIO3
#define GREEN_LED  D3  // GPIO4
#define WHITE_LED  D10  // GPIO9

// Button pin
#define BUTTON_PIN D9

// Sampling config
#define SAMPLE_RATE_MS 10
#define CAPTURE_DURATION_MS 1000
#define FEATURE_SIZE EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE

// Data capture state
bool capturing = false;
unsigned long last_sample_time = 0;
unsigned long capture_start_time = 0;
int sample_count = 0;

// Feature buffer
float features[FEATURE_SIZE];

// Required by Edge Impulse classifier
int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

/**
 * Setup WiFi connection
*/
void setupWiFi() {
    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("");
    Serial.print("Connected to ");
    Serial.println(ssid);
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

void setup()
{
    Serial.begin(115200);
    Wire.begin(6, 7);  // SDA = GPIO6, SCL = GPIO7

    // LED setup
    pinMode(YELLOW_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);
    pinMode(WHITE_LED, OUTPUT);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(WHITE_LED, LOW);

    // Button setup
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // MPU init
    Serial.println("Initializing MPU6050...");
    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) { delay(10); }
    }

    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    setupWiFi();

    Serial.println("=== Magic Wand Ready ===");
    Serial.printf("Confidence threshold: %.1f%%\n", CONFIDENCE_THRESHOLD);
    Serial.println("Send 'o' to start inference or press button");
}

void capture_accelerometer_data() {
    if (millis() - last_sample_time >= SAMPLE_RATE_MS) {
        last_sample_time = millis();

        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);

        if (sample_count < FEATURE_SIZE / 3) {
            int idx = sample_count * 3;
            features[idx] = a.acceleration.x;
            features[idx + 1] = a.acceleration.y;
            features[idx + 2] = a.acceleration.z;
            sample_count++;
        }

        if (millis() - capture_start_time >= CAPTURE_DURATION_MS) {
            capturing = false;
            run_inference();
        }
    }
}

void run_inference() {
    Serial.println("\n=== GESTURE INFERENCE RESULT ===");
    
    ei_impulse_result_t result = { 0 };

    signal_t features_signal;
    features_signal.total_length = FEATURE_SIZE;
    features_signal.get_data = &raw_feature_get_data;

    EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);
    if (res != EI_IMPULSE_OK) {
        Serial.print("Classifier failed: "); Serial.println(res);
        return;
    }

    // Find the gesture with highest confidence
    int max_index = -1;
    float max_value = 0.0;
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > max_value) {
            max_value = result.classification[i].value;
            max_index = i;
        }
    }

    // Display results clearly
    const char* predicted_gesture = ei_classifier_inferencing_categories[max_index];
    float confidence_percentage = max_value * 100.0;
    
    Serial.printf("Predicted Gesture: %s\n", predicted_gesture);
    Serial.printf("Confidence Level: %.1f%%\n", confidence_percentage);
    Serial.printf("Threshold: %.1f%%\n", CONFIDENCE_THRESHOLD);
    
    // Edge-Cloud Offloading Decision
    if (confidence_percentage < CONFIDENCE_THRESHOLD) {
        Serial.println("Decision: LOW CONFIDENCE - Offloading to cloud server");
        Serial.println("Action: Sending raw sensor data to server for inference");
        sendRawDataToServer();
    } else {
        Serial.println("Decision: HIGH CONFIDENCE - Processing locally");
        Serial.println("Action: Using local inference result");
        processLocalInference(predicted_gesture);
    }
    
    Serial.println("=====================================\n");
}

void processLocalInference(const char* gesture) {
    // Reset all LEDs
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(WHITE_LED, LOW);

    Serial.printf("Local Processing: Gesture '%s'\n", gesture);
    
    // Control LEDs based on gesture
    if (strcmp(gesture, "O") == 0) {
        digitalWrite(WHITE_LED, HIGH);
        Serial.println("LED Action: WHITE LED ON");
    } 
    else if (strcmp(gesture, "V") == 0) {
        digitalWrite(GREEN_LED, HIGH);
        Serial.println("LED Action: GREEN LED ON");
    } 
    else if (strcmp(gesture, "Z") == 0) {
        Serial.println("LED Action: GREEN/YELLOW BLINKING");
        for (int i = 0; i < 3; i++) {
            digitalWrite(GREEN_LED, HIGH);
            digitalWrite(YELLOW_LED, LOW);
            delay(150);
            digitalWrite(GREEN_LED, LOW);
            digitalWrite(YELLOW_LED, HIGH);
            delay(150);
        }
        digitalWrite(GREEN_LED, LOW);
        digitalWrite(YELLOW_LED, LOW);
    }
    else {
        Serial.println("LED Action: Unknown gesture - no LED action");
    }
}

void sendRawDataToServer() {
    HTTPClient http;
    http.begin(serverUrl);
    http.setTimeout(10000); // 10 second timeout
    http.addHeader("Content-Type", "application/json");

    // Build JSON payload with raw sensor data
    String jsonPayload = "{";
    jsonPayload += "\"student_id\":\"";
    jsonPayload += studentId;
    jsonPayload += "\",";
    jsonPayload += "\"data\":[";
    
    for (int i = 0; i < FEATURE_SIZE; i++) {
        jsonPayload += String(features[i], 4);
        if (i < FEATURE_SIZE - 1) {
            jsonPayload += ",";
        }
    }
    jsonPayload += "]}";

    Serial.println("Sending raw IMU data to cloud server...");
    Serial.printf("Data size: %d features\n", FEATURE_SIZE);

    int httpResponseCode = http.POST(jsonPayload);
    Serial.printf("HTTP Response Code: %d\n", httpResponseCode);

    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.println("Server Response: " + response);

        // Parse server response
        DynamicJsonDocument doc(256);
        DeserializationError error = deserializeJson(doc, response);
        if (!error) {
            const char* server_gesture = doc["gesture"];
            float server_confidence = doc["confidence"];
            
            Serial.printf("Server Inference Result:\n");
            Serial.printf("  Gesture: %s\n", server_gesture);
            Serial.printf("  Confidence: %.1f%%\n", server_confidence * 100.0);
            
            // Use server result for LED control
            processServerInference(server_gesture);
        } else {
            Serial.println("Failed to parse server response");
        }
    } else {
        Serial.printf("HTTP Request Failed: %s\n", http.errorToString(httpResponseCode).c_str());
        Serial.println("Fallback: No LED action due to server connection failure");
    }

    http.end();
}

void processServerInference(const char* gesture) {
    // Reset all LEDs
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(WHITE_LED, LOW);

    Serial.printf("Server Processing: Gesture '%s'\n", gesture);
    
    // Control LEDs based on server result
    if (strcmp(gesture, "O") == 0) {
        digitalWrite(WHITE_LED, HIGH);
        Serial.println("LED Action: WHITE LED ON (from server)");
    } 
    else if (strcmp(gesture, "V") == 0) {
        digitalWrite(GREEN_LED, HIGH);
        Serial.println("LED Action: GREEN LED ON (from server)");
    } 
    else if (strcmp(gesture, "Z") == 0) {
        Serial.println("LED Action: GREEN/YELLOW BLINKING (from server)");
        for (int i = 0; i < 3; i++) {
            digitalWrite(GREEN_LED, HIGH);
            digitalWrite(YELLOW_LED, LOW);
            delay(150);
            digitalWrite(GREEN_LED, LOW);
            digitalWrite(YELLOW_LED, HIGH);
            delay(150);
        }
        digitalWrite(GREEN_LED, LOW);
        digitalWrite(YELLOW_LED, LOW);
    }
    else {
        Serial.println("LED Action: Unknown server gesture - no LED action");
    }
}

void loop() {
    // Trigger via button (LOW means pressed)
    if (digitalRead(BUTTON_PIN) == LOW && !capturing) {
        Serial.println("Button pressed! Starting gesture capture...");
        sample_count = 0;
        capturing = true;
        capture_start_time = millis();
        last_sample_time = millis();
        delay(200);  // debounce
    }

    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd == 'o') {
            sample_count = 0;
            capturing = true;
            capture_start_time = millis();
            last_sample_time = millis();
            Serial.println("Starting gesture capture...");
        }
    }

    if (capturing) {
        capture_accelerometer_data();
    }
}