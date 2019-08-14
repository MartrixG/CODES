#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
struct d
{
    int x, y;
} we[1000010];
int n;
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d%d", &we[i].x, &we[i].y);
    }
}