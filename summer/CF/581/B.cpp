#include <cstdio>
int main()
{
    int n, l, r;
    scanf("%d%d%d", &n, &l, &r);
    int mi = 0, ma = 0;
    mi = n - l + (1 << l) - 1;
    ma = (n - r) * (1 << (r - 1)) + (1 << r) - 1;
    printf("%d %d\n", mi, ma);
}