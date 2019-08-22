#include <cstdio>
int t, b, p, f;
int h, c;
int min(int a, int b) { return a < b ? a : b; }
int main()
{
    scanf("%d", &t);
    for (int i = 1; i <= t; i++)
    {
        scanf("%d%d%d%d%d", &b, &p, &f, &h, &c);
        int ans = 0;
        int num1, num2;
        if(h>c)
        {
            num1 = min(b / 2, p);
            ans += num1 * h;
            b -= 2 * num1;
            num2 = min(b / 2, f);
            ans += num2 * c;            
        }
        else
        {
            num1 = min(b / 2, f);
            ans += num1 * c;
            b -= 2 * num1;
            num2 = min(b / 2, p);
            ans += num2 * h;
        }
        printf("%d\n", ans);
    }
}