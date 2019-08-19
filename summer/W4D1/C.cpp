#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
using namespace std;
string s;
int f[110][110];
int pd(char a, char b)
{
    if (a == '(' && b == ')')
        return 1;
    if (a == '[' && b == ']')
        return 1;
    return 0;
}
int max(int a, int b) { return a > b ? a : b; }
int main()
{
    while (cin >> s)
    {
        if (s[0] == 'e')
            return 0;
        memset(f, 0, sizeof(f));
        int n = s.size();
        for (int tmp = 1; tmp <= n; tmp++)
        {
            for (int i = 0; i + tmp - 1 < n; i++)
            {
                int j = i + tmp - 1;
                if (pd(s[i], s[j]))
                {
                    f[i][j] = f[i + 1][j - 1] + 2;
                }
                for (int k = i; k <= j; k++)
                {
                    f[i][j] = max(f[i][j], f[i][k] + f[k][j]);
                }
            }
        }
        printf("%d\n", f[0][n - 1]);
    }
}