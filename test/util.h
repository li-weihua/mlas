#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>

template <typename T>
std::vector<T> ReadRawData(const std::string& file_name) {
  std::ifstream fp(file_name, std::ios::binary | std::ios::ate);
  if (!fp.is_open()) {
    std::cerr << "cannot open file: " << file_name << std::endl;
    std::abort();
  }

  size_t size = fp.tellg();
  fp.seekg(0, std::ios::beg);

  int n = size / sizeof(T);

  std::vector<T> v(n);

  fp.read(reinterpret_cast<char*>(v.data()), size);

  return v;
}
