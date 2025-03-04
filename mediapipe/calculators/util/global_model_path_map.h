#include <map>
#include <string>

namespace mediapipe {
class GlobalModelPathMap {
 private:
  static std::map<std::string, std::string> data;

 public:
  static std::map<std::string, std::string> Get();
  static void Set(std::map<std::string, std::string> new_data);
  static void Add(std::string key, std::string value);
  static void Remove(std::string key);
  static void Clear();
  static void GetRealModelPath(std::string& path);
};
}  // namespace mediapipe