#include <stdlib.h>

#include "conv2d.h"
#include "common.h"
#include "util.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " data-path" << std::endl;
    return 1;
  }

  Conv2d conv2d;

  size_t buffer_size = conv2d.GetBufferSize();

  void* buffer = aligned_alloc(32, sizeof(float) * buffer_size);

  conv2d.SetBuffer(buffer);

  float* output = (float*)aligned_alloc(32, sizeof(float) * 1 * 48 * 1 * 40);

  std::string path(argv[1]);

  auto input = ReadRawData<float>(path + "/input.data");
  auto ref_output = ReadRawData<float>(path + "/output.data");
  auto weight = ReadRawData<float>(path + "/weight.data");
  auto bias = ReadRawData<float>(path + "/bias.data");

  // do forward
  // conv2d.DoForward(output, input, kernel, bias);
  conv2d.DoForward(output, input.data(), weight.data(), bias.data());

  float r = get_max_diff(output, ref_output.data(), 48 * 40);

  std::cout << "max abs diff: " << r << std::endl;

  // free buffer
  free(buffer);

  return 0;
}
