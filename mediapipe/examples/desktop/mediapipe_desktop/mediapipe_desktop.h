#pragma once

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK void initLogging(const char *argv0);

EXPORT_SDK void shutdownGoogleLogging();

#undef EXPORT_SDK