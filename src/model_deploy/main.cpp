#include "accelerometer_handler.h"

#include "config.h"

#include "magic_wand_model_data.h"


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/version.h"

#include "uLCD_4DGL.h"

#include "mbed_rpc.h"

#include "stm32l475e_iot01_accelero.h"

#include "MQTTNetwork.h"

#include "MQTTmbed.h"

#include "MQTTClient.h"


// Create an area of memory to use for input, output, and intermediate arrays.

// The size of this will depend on the model you're using, and may need to be

// determined by experimentation.

constexpr int kTensorArenaSize = 60 * 1024;

uint8_t tensor_arena[kTensorArenaSize];

uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

// Return the result of the last prediction

int i = 30, mode = 0;
DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
DigitalOut myled3(LED3);
BufferedSerial pc(USBTX, USBRX);
void ModeControl(Arguments *in, Reply *out);
RPCFunction rpcLED(&ModeControl, "ModeControl");
double x, y;
Thread thread1;
Thread thread2;
Thread thread3;
int angle;

WiFiInterface *wifi;

InterruptIn btn2(USER_BUTTON);

//InterruptIn btn3(SW3);

volatile int message_num = 0;

volatile int arrivedcount = 0;

volatile bool closed = false;


const char* topic = "Mbed";


Thread mqtt_thread(osPriorityHigh);

EventQueue mqtt_queue;


void messageArrived(MQTT::MessageData& md) {

    MQTT::Message &message = md.message;

    char msg[300];

    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);

    printf(msg);

    ThisThread::sleep_for(1000ms);

    char payload[300];

    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);

    printf(payload);

    ++arrivedcount;

}


void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client) {

    //message_num++;

    MQTT::Message message;

    char buff[100];

    if (mode == 1) {
      sprintf(buff, " %d %d ", mode, i);
      uLCD.locate(1, 6);
      message_num = 0;
    }
    if (mode == 2) {
      sprintf(buff, " %d %d %d ", mode, angle, message_num++);
    }

    message.qos = MQTT::QOS0;

    message.retained = false;

    message.dup = false;

    message.payload = (void*) buff;

    message.payloadlen = strlen(buff) + 1;

    int rc = client->publish(topic, message);


    //printf("rc:  %d\r\n", rc);

    //printf("Puslish message: %s\r\n", buff);

}


void close_mqtt() {

    closed = true;

}

int PredictGesture(float* output) {

  // How many times the most recent gesture has been matched in a row

  static int continuous_count = 0;

  // The result of the last prediction

  static int last_predict = -1;


  // Find whichever output has a probability > 0.8 (they sum to 1)

  int this_predict = -1;

  for (int i = 0; i < label_num; i++) {

    if (output[i] > 0.8) this_predict = i;

  }

// No gesture was detected above the threshold

  if (this_predict == -1) {

    continuous_count = 0;

    last_predict = label_num;

    return label_num;

  }


  if (last_predict == this_predict) {

    continuous_count += 1;

  } else {

    continuous_count = 0;

  }

  last_predict = this_predict;


  // If we haven't yet had enough consecutive matches for this gesture,

  // report a negative result

  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {

    return label_num;

  }

  // Otherwise, we've seen a positive result, so clear all our variables

  // and report it

  continuous_count = 0;

  last_predict = -1;


  return this_predict;

}

void rpc_thread();
int ui_thread();
int angle_thread();

int main(int argc, char* argv[]) {
  uLCD.locate(1, 2);
  uLCD.printf("%2d", i);

  BSP_ACCELERO_Init();

  thread1.start(rpc_thread);
  thread2.start(ui_thread);
  thread3.start(angle_thread);

}

void rpc_thread() {
    while(1) {
     char buf[256], outbuf[256];
      FILE *devin = fdopen(&pc, "r");
      FILE *devout = fdopen(&pc, "w");

    memset(buf, 0, 256);
    for (int i = 0; ; i++) {
        char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
    }
    //Call the static call method on the RPC class
    RPC::call(buf, outbuf);
    printf("%s\r\n", outbuf);
    }
}

int ui_thread(){
  // Whether we should clear the buffer next time we fetch data

  bool should_clear_buffer = false;

  bool got_data = false;


  // The gesture index of the prediction

  int gesture_index;


  // Set up logging.

  static tflite::MicroErrorReporter micro_error_reporter;

  tflite::ErrorReporter* error_reporter = &micro_error_reporter;


  // Map the model into a usable data structure. This doesn't involve any

  // copying or parsing, it's a very lightweight operation.

  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {

    error_reporter->Report(

        "Model provided is schema version %d not equal "

        "to supported version %d.",

        model->version(), TFLITE_SCHEMA_VERSION);

    return -1;

  }


  // Pull in only the operation implementations we need.

  // This relies on a complete list of all the ops needed by this graph.

  // An easier approach is to just use the AllOpsResolver, but this will

  // incur some penalty in code space for op implementations that are not

  // needed by this graph.

  static tflite::MicroOpResolver<6> micro_op_resolver;

  micro_op_resolver.AddBuiltin(

      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                               tflite::ops::micro::Register_MAX_POOL_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                               tflite::ops::micro::Register_CONV_2D());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                               tflite::ops::micro::Register_FULLY_CONNECTED());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                               tflite::ops::micro::Register_SOFTMAX());

  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,

                               tflite::ops::micro::Register_RESHAPE(), 1);


  // Build an interpreter to run the model with

  static tflite::MicroInterpreter static_interpreter(

      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

  tflite::MicroInterpreter* interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors

  interpreter->AllocateTensors();


  // Obtain pointer to the model's input tensor

  TfLiteTensor* model_input = interpreter->input(0);

  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;

  }


  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

  if (setup_status != kTfLiteOk) {

    error_reporter->Report("Set up failed\n");

    return -1;

  }


  error_reporter->Report("Set up successful...\n");


  while (true) {


    // Attempt to read new data from the accelerometer
    if (mode == 1) {

    got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                 input_length, should_clear_buffer);


    // If there was no new data,

    // don't try to clear the buffer again and wait until next time

    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error

    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction

    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data

    should_clear_buffer = gesture_index < label_num;
    }

    // Produce an output

    if(mode == 1) {
      myled1 = 1;
      myled2 = 0;

      if (gesture_index < label_num) {
        if (gesture_index == 1) {
          i = i + 5;
          uLCD.locate(1, 1);
          uLCD.printf("slope");
          uLCD.locate(1, 2);
          uLCD.printf("%2d", i);
        } else {
          i = i - 5;
          uLCD.locate(1, 1);
          uLCD.printf("ring ");
          uLCD.locate(1, 2);
          uLCD.printf("%2d", i);
        }
      }
    }
  }
}

void ModeControl (Arguments *in, Reply *out)   {

    bool success = true;
    // In this scenario, when using RPC delimit the two arguments with a space.
    x = in->getArg<double>();
    //y = in->getArg<double>();

    if (x == 1) {
      mode = 1;
      myled1 = 1;
      myled2 = 0;
    } else if (x == 2) {
      mode = 2;
      myled1 = 0;
      myled2 = 0;
    } else {
      mode = 0;
      myled1 = 0;
      myled2 = 1;
    }

   char buffer[20], outbuf[20];

    char strings[20];

    RPC::call(buffer, outbuf);

    if (success) {

        out->putData(buffer);

    } else {

        out->putData("Failed to execute LED control.");

    }
}

int angle_thread() {
  bool init = false;
  int sumx = 0, sumy = 0, sumz = 0; 
  int sum;
  double lengthA, lengthB;
  double shit;



wifi = WiFiInterface::get_default_instance();

    if (!wifi) {

            printf("ERROR: No WiFiInterface found.\r\n");

            return -1;

    }

    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);

    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);

    if (ret != 0) {

            printf("\nConnection error: %d\r\n", ret);

            return -1;

    }

    NetworkInterface* net = wifi;

    MQTTNetwork mqttNetwork(net);

    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);


    //TODO: revise host to your IP

    const char* host = "192.168.43.135";

    printf("Connecting to TCP network...\r\n");


    SocketAddress sockAddr;

    sockAddr.set_ip_address(host);

    sockAddr.set_port(1883);


    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting


    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);

    if (rc != 0) {

            printf("Connection error.");

            return -1;

    }

    printf("Successfully connected!\r\n");


    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;

    data.MQTTVersion = 3;

    data.clientID.cstring = "Mbed";


    if ((rc = client.connect(data)) != 0){

            printf("Fail to connect MQTT\r\n");

    }

    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){

            printf("Fail to subscribe\r\n");

    }


    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));

    btn2.rise(mqtt_queue.event(&publish_message, &client));

    //btn3.rise(&close_mqtt);

    int num = 0;

  while(1){
    myled3 = 0;
    while (mode == 2) {
      myled3 = 1;
      int16_t pDataXYZ[3] = {0};

          char buffer[200];
          if (init == false) {
            myled1 = 1;
            myled2 = 1;
            for (int j = 0; j < 10; j++) {
              BSP_ACCELERO_AccGetXYZ(pDataXYZ);
              sumx += pDataXYZ[0];
              sumy += pDataXYZ[1];
              sumz += pDataXYZ[2];
              ThisThread::sleep_for(20ms);
            }

            sumx = sumx / 10;
            sumy = sumy / 10;
            sumz = sumz / 10;
            init = true;
          }


            BSP_ACCELERO_AccGetXYZ(pDataXYZ);
            sum = sumx * pDataXYZ[0] + sumy * pDataXYZ[1] + sumz * pDataXYZ[2];
            lengthA = sqrt(sumx*sumx+sumy*sumy+sumz*sumz);
            lengthB = sqrt(pDataXYZ[0]*pDataXYZ[0]+pDataXYZ[1]*pDataXYZ[1]+pDataXYZ[2]*pDataXYZ[2]);
            shit = 1.0*sum / lengthA / lengthB;
            angle = acos(shit) * 57.4;

            uLCD.locate(1, 3);
            uLCD.printf("%2d", angle);

            if (angle > i) {
              mqtt_queue.call(&publish_message, &client);
            }
            ThisThread::sleep_for(100ms);

    //sprintf(buffer, "Accelerometer values: (%d, %d, %d)", pDataXYZ[0], pDataXYZ[1], pDataXYZ[2]);
    //printf("%s\n", buffer);
    }
  }
}