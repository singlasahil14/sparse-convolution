## sparse-convolution
tensorflow op for sparse-convolution 

## Requirements
tensorflow r1.3

## Run the code
```bash
g++ -std=c++11 -shared sparse_conv.cc -o sparse_conv.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
python sparse_conv_test.py
```
