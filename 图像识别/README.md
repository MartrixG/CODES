ResNet
<font size = 2>
|layer_name |output_size |18-layer|34-layer|38-layer|
|:-:        |:-:         |:-:     |:-:|:-:|
|conv1|112 x 112|7 x 7, 64, stride 2|7 x 7, 64, stride 2|3 x 3,16,stride 1
|conv2| 56 x 56| $ \left[\begin{matrix} 3 * 3, & 64 \\ 3 * 3,&64\end{matrix}\right]*2 $ | $$ \left[\begin{matrix} 3 * 3, & 64 \\ 3 * 3,&64\end{matrix}\right] * 3 $$ | $$ \left[\begin{matrix} 3 * 3, & 16 \\ 3 * 3,&16\end{matrix}\right] * 6 $$
|conv3_x|28 x 28|$$\left[\begin{matrix} 3 * 3, & 128 \\ 3 * 3,&128\end{matrix}\right]*2 $$ | $$ \left[\begin{matrix} 3 * 3, & 128 \\ 3 * 3,&128\end{matrix}\right] * 4 $$ | $$ \left[\begin{matrix} 3 * 3, & 32 \\ 3 * 3,&32\end{matrix}\right] * 6 $$
|conv4_x|14 x 14|$$\left[\begin{matrix} 3 * 3, & 256 \\ 3 * 3,&256\end{matrix}\right]*2 $$ | $$ \left[\begin{matrix} 3 * 3, & 256 \\ 3 * 3,&256\end{matrix}\right] * 6 $$ | $$ \left[\begin{matrix} 3 * 3, & 64 \\ 3 * 3,&64\end{matrix}\right] * 6 $$
|conv5_x|7 x 7|$$\left[\begin{matrix} 3 * 3, & 512 \\ 3 * 3,&512\end{matrix}\right]*2 $$ | $$ \left[\begin{matrix} 3 * 3, & 512 \\ 3 * 3,&512\end{matrix}\right] * 3 $$
| |1 x 1|average pool, 1000-d, softmax|average pool, 1000-d, softmax|average pool, 10-d, softmax
</font>

MobileNet
<font size = 2>
layer_name | output_size | layer
:-:|:-:|:-:
conv_1|16 x 16|$$ \left[\begin{matrix} 3 * 3, & 16\end{matrix}\right] * 6 $$
conv_2|8 x 8|$$ \left[\begin{matrix} 3 * 3, & 32\end{matrix}\right] * 5 $$
conv_3|4 x 4|$$ \left[\begin{matrix} 3 * 3, & 32\end{matrix}\right] * 4$$
| |1 x 1|average pool, 10-d, softmax
</font>

InceptionNet
<font size = 2>
layer_name | output_size | layer |
:-:|:-:|:-:
conv_1|16 x 32 x 32| 3 x 3, 16, stride 1
inception(1a)|32 x 32 x 32|$$ \left[\begin{matrix} 1 * 1, & 32\end{matrix}\right]$$
inception(1b)|32 x 32 x 32|$$ \left[\begin{matrix} 1 * 1, & 8 \\ 3*3, &32\end{matrix}\right]$$
|inception(1c)|24 x 32 x 32|$$ \left[\begin{matrix} 1 * 1, & 8 \\ 5*5, &24\end{matrix}\right]$$
|incetion(1d)|32 x 32 x 32|maxpool 3 x 3$$ \left[\begin{matrix} 1 * 1, & 32 \end{matrix}\right]$$
|conv_2|120 x 16 x 16|inception(1) x 2, maxpool 3 x 3 stride 2 |
inception(2a)|48 x 16 x 16|$$ \left[\begin{matrix} 1 * 1, & 48\end{matrix}\right]$$
inception(2b)|48 x 16 x 16|$$ \left[\begin{matrix} 1 * 1, & 16 \\ 3*3, &48\end{matrix}\right]$$
|inception(2c)|32 x 16 x 16|$$ \left[\begin{matrix} 1 * 1, & 16 \\ 5*5, &32\end{matrix}\right]$$
|incetion(2d)|48 x 16 x 16|maxpool 3 x 3$$ \left[\begin{matrix} 1 * 1, & 48 \end{matrix}\right]$$
|conv_3|176 x 8 x 8|inception(2) x 5, maxpool 3 x 3 stride 2 |
inception(2a)|64 x 8 x 8|$$ \left[\begin{matrix} 1 * 1, & 48\end{matrix}\right]$$
inception(2b)|64 x 8 x 8|$$ \left[\begin{matrix} 1 * 1, & 32 \\ 3*3, &64\end{matrix}\right]$$
|inception(2c)|48 x 8 x 8|$$ \left[\begin{matrix} 1 * 1, & 32 \\ 5*5, &48\end{matrix}\right]$$
|incetion(2d)|64 x 8 x 8|maxpool 3 x 3$$ \left[\begin{matrix} 1 * 1, & 64 \end{matrix}\right]$$
|conv_4|240 x 4 x 4|inception(2) x 2, average pool 4 x 4 |
| |1 x 1|average pool, 10-d, softmax
</font>

DenseNet
<font size = 2>
|layer_name |output_size |121-layer|58-layer|
|:-:        |:-:         |:-:     |:-:
|conv1|112 x 112|7 x 7, 64, stride 2|3 x 3, 16, stride 1|
|block1| 56 x 56|$$\left[\begin{matrix} 1 * 1\\ 3 * 3\end{matrix}\right]*6 $$ averagepool 2 x 2 stride 2| $$ \left[\begin{matrix} 1 * 1\\ 3 * 3\end{matrix}\right] * 9 $$ averagepool 2 x 2 stride 2
|block2|28 x 28|$$\left[\begin{matrix} 1 * 1 \\ 3 * 3\end{matrix}\right]*12 $$  averagepool 2 x 2 stride 2| $$ \left[\begin{matrix} 1 * 1\\ 3 * 3\end{matrix}\right] * 9 $$ averagepool 2 x 2 stride 2
|block3|14 x 14|$$\left[\begin{matrix} 1 * 1 \\ 3 * 3\end{matrix}\right]*24 $$  averagepool 2 x 2 stride 2| $$ \left[\begin{matrix} 1 * 1\\ 3 * 3\end{matrix}\right] * 9 $$ 
|block4|7 x 7|$$\left[\begin{matrix} 1 * 1 \\ 3 * 3\end{matrix}\right]*16 $$|
| |1 x 1|average pool, 1000-d, softmax|average pool, 10-d, softmax
</font>