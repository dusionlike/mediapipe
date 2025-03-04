#include "mediapipe/calculators/util/global_model_path_map.h"

namespace mediapipe {
std::map<std::string, std::string> GlobalModelPathMap::data;
std::map<std::string, std::string> GlobalModelPathMap::Get() { return data; }
void GlobalModelPathMap::Set(std::map<std::string, std::string> new_data) {
  data = new_data;
}
void GlobalModelPathMap::Add(std::string key, std::string value) {
  data[key] = value;
}
void GlobalModelPathMap::Remove(std::string key) { data.erase(key); }
void GlobalModelPathMap::Clear() { data.clear(); }
void GlobalModelPathMap::GetRealModelPath(std::string& path) {
  if (data.find(path) != data.end()) {
    path = data[path];
  }
}
}  // namespace mediapipe