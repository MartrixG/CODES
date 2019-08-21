#include <cstdio>
int f[1000];
int find(int x)
{
    if (x != f[x])
        f[x] = find(f[x]);
    else
        return x;
}
int link(int x, int y)
{
    int fx = find(x), fy = find(y);
    f[fx] = fy;
}
int main()
{
    for (int i = 1; i <= 100; i++)
        f[i] = i;
}
//并查集