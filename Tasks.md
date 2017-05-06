1. Tim training / test data (MIT saliency images data set) va dataset ma paper(s) da dung
2. Quyet dinh programming language (Python for git, C++ for class) + OpenCV
3. List paper for implementation:
- "Salient Region Detection by Modeling Distributions of Color and Orientation", Viswanath et al. (CSF and OSF)
- ...
4. Implement CSF:
Luu y khi implement random cua numpy: phai set seed 2017 (de reproduce duoc). Neu duoc thi tao them bo test
Có 7 bước chính:
- 1) Tim dominant hue
- 2) Initilize va chay EM de ra GMM (ket thuc o Fig 2)
- 3) Tinh các thông số xác suất (công thức 5, 6, 7, 8)
- 4) Tinh Compactness
- 5) Tinh Isolation
- 6) Tinh pixel saliency
- 7) Chay tren bo test va evaluate theo paper do

Deadline: 31/5
