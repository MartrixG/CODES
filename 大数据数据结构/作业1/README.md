<center style="font-size:30px">2020春《大数据高级数据结构设计与实践》作业一 </center><br>
<center>主讲教师: 史建焘</center><br>
<center>姓名: <u>许天骁</u>  学号: <u>1171000405</u>   班号: <u>1703109</u></center>


##### bitmap
1. 简介
   所谓的Bit-map就是用一个bit位来标记某个元素对应的Value， 而Key即是该元素。可以理解成一个位图是一个巨大的储存数据的桶，桶的下标表示元素，而每个桶中只保存一个比特位，若为1则表示该元素存在，若为0则表示该元素不存在。这样可以大大减少使用的空间。
2. 实现原理
   可以使用字符串数组来表示所有的桶，一个字符出所占空间是一个字节，8bit。所以通过给定的上下界可以确定桶的大小。除以8则是需要申请的字符串数组的大小。所有位均初始化为0。
##### bloomfilter
1. 简介
   结合了位图和哈希表的优点，位图的优点是节省空间，但是只能处理整型值一类的问题，无法处理字符串一类的问题。而Hash表却恰巧解决了位图无法解决的问题，然而Hash太浪费空间。bloomfilter是一种基于二进制向量和一系列随机函数的数据结构。
2. 实现原理
   如果要查找某个元素item是否在S中，还是通过映射函数{f1,f2.....fk}得到k个值{g1,g2.....gk}。然后再判断array[g1],array[g2]......array[gk]是否都为1，若全为1，则item在S中，否则item不在S中。

##### 实验任务及实现
1. 任务1
   1 ：生成1万个随机数 （srand  rand函数 随机数范围0到rand_max）
2 ：用bitmap表示这1万个随机数
 3：将1万个随机数放到bloomfilter容器中，错误率不高于0.001
 4：再产生1000个随机数，通过bitmap和bloomfilter判断，检测bloomfilter的误判率
1. 任务2
     1：判断 file2中url总数，设定错误率为0.001；
   2：将file2中的url放入到bloomfilter容器中；
   3：对file1中的每一个url进行判断是否在bloomfilter容器中。
   4：输出所有找到的url


3. 分析
    ![](1.png)
 在10000数据量上，对布隆过滤器的参数进行设置，设置集合大小为10000。1000个随机数的监测，错误率为0。
  ![](2.png)
 首先浏览file2文件，设置布隆过滤器的集合大小为470000.统计结果是461281条url。并且将所有在file1的url和在file2中进行比对。若在file2文件中出现，则进行输出，输出至count.txt文件中。大概有30000条url被统计在内。
 通过调试程序，查看任务管理器，可以看到两个程序的内存使用都十分少。
##### 源代码
任务1
```c++
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;
#include "bitmap.hpp"
#include "bloom_filter.hpp"

int a[10000];
int rand_max = 20000;

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < 10000; i++)
    {
        a[i] = rand() % rand_max;
    }
    bloom_parameters parameters;
    parameters.projected_element_count = 10000;
    parameters.false_positive_probability = 0.001;
    parameters.random_seed = 0xC7C7C7C7;
    if (!parameters)
    {
        cout << "Error - Invalid set of bloom filter parameters!" << endl;
        return 1;
    }
    parameters.compute_optimal_parameters();
    bloom_filter filter(parameters);
    BitMap bitmap(0, rand_max);
    for (int i = 0; i < 10000; i++)
    {
        bitmap.set(a[i]);
        filter.insert(a[i]);
    }
    int wrong = 0;
    for (int i = 0; i < 1000; i++)
    {
        int tmp = rand() % rand_max;
        int flag_bit = 0, flag_bloom = 0;
        if (bitmap.test(tmp))
        {
            flag_bit = 1;
        }
        if (filter.contains(tmp))
        {
            flag_bloom = 1;
        }
        if (flag_bit != flag_bloom)
        {
            wrong++;
        }
    }
    cout << "wrong rate:" << double(wrong) / 1000.0 << endl;
}
```

任务2
``` c++
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
using namespace std;
#include "bloom_filter.hpp"
void readfile2(bloom_filter &filter)
{
    ifstream file2("file2.txt");
    string line;
    while (getline(file2, line))
    {
        filter.insert(line);
    }
    cout << "file2 contains " << filter.element_count() << " urls." << endl;
    return;
}
void test(bloom_filter &filter)
{
    ifstream file1("file1.txt");
    ofstream outfile("count.txt");
    string line;
    int count = 0;
    while (getline(file1, line))
    {
        if (filter.contains(line))
        {
            outfile << line << endl;
        }
    }
    outfile.close();
}
int main()
{

    bloom_parameters parameters;
    parameters.projected_element_count = 470000;
    parameters.false_positive_probability = 0.001;
    parameters.random_seed = 0xC7C7C7C7;
    if (!parameters)
    {
        cout << "Error - Invalid set of bloom filter parameters!" << endl;
        return 1;
    }
    parameters.compute_optimal_parameters();
    bloom_filter filter(parameters);
    readfile2(filter);
    test(filter);
}
```