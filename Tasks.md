1. Tim training / test data (MIT saliency images data set)
2. Quyet dinh programming language (Python for git, C++ for class) + OpenCV
3. List paper for implementation:
- "Salient Region Detection by Modeling Distributions of Color and Orientation", Viswanath et al. (CSF and OSF)
- ...
4. Implement CSF:
Luu y khi implement random cua numpy: phai set seed 2017 (de reproduce duoc). Neu duoc thi tao them bo test
Có 5 bước:
- Tim dominant hue
- Initilize va chay EM de ra GMM (ket thuc o Fig 2)
- Tinh các thông số xác suất (công thức 5, 6, 7, 8)
- 
