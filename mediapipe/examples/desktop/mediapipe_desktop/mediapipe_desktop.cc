#include "mediapipe_desktop.h"

#include <glog/logging.h>

void initLogging(const char *argv0) {
  google::InitGoogleLogging(argv0);
}

void shutdownGoogleLogging() {
  google::ShutdownGoogleLogging();
}