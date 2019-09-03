#include <cstdio>
#include <cstring>
#include <string>
#include <queue>
#include <iostream>
using namespace std;
int T, n;
queue<int> th[10001];
queue<char> op[10001];
int main()
{
    scanf("%d%d", &T, &n);
    cin.get();
    while (T--)
    {
        for (int i = 0; i < n; i++)
        {
            while (!th[i].empty())
                th[i].pop();
            while (!op[i].empty())
                op[i].pop();
        }
        for (int i = 0; i < n; i++)
        {
            char tmp[60];
            cin.getline(tmp, 60);
            int l = strlen(tmp);
            int k = 0;
            while (k < l)
            {
                op[i].push(tmp[k]);
                int num = 0;
                k++;
                while (tmp[k] != ' ' && k < l)
                {
                    char sss = tmp[k];
                    num *= 10;
                    num += tmp[k] - '0';
                    k++;
                }
                k++;
                th[i].push(num);
            }
        }
        int f0 = 0;
        int f = 1;
        while (f)
        {
            f = 0;
            for (int i = 0; i < n; i++)
            {
                int th1 = i, th2 = th[i].front();
                if (op[th1].size() == 0 || op[th2].size() == 0)
                    continue;
                char op1 = op[th1].front(), op2 = op[th2].front();
                if ((op1 == 'R' && op2 == 'S') || (op1 == 'S' && op2 == 'R'))
                {
                    if (th1 == th[th2].front())
                    {
                        th[th1].pop(), op[th1].pop();
                        th[th2].pop(), op[th2].pop();
                        f = 1;
                    }
                }
                if ((op1 == 'R' && op2 == 'R') || (op1 == 'S' && op2 == 'S'))
                {
                    if (th1 == th[th2].front())
                    {
                        f = 0;
                        f0 = 1;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++)
            if (op[i].size() != 0)
                f0 = 1;
        printf("%d\n", f0);
    }
}
/*
2 3
R1 S1
R2 S0 R0 S2
S1 R1
R1
R2 S0 R0
S1 R1

3 2
R1 S1
S0 R0
R1 S1
R0 S0
R1 R1 R1 R1 S1 S1 S1 S1
S0 S0 S0 S0 R0 R0 R0 R0
*/