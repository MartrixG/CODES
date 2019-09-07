#include <cstdio>
#include <cstring>
#include <string>
#include <stack>
#include <iostream>
using namespace std;
void work()
{
    string s;
    cin >> s;
    stack<int> num;
    stack<char> op;
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] >= '0' && s[i] <= '9')
        {
            if (num.size() == 0)
            {
                num.push(s[i] - '0');
                continue;
            }
            int num1, num2;
            num1 = num.top(), num.pop();
            num2 = s[i] - '0';
            if (op.top() == 'x')
            {
                num.push(num1 * num2);
                op.pop();
            }
            else if (op.top() == '/')
            {
                num.push(num1 / num2);
                op.pop();
            }
            else
            {
                num.push(num1);
                num.push(num2);
            }
        }
        else
        {
            op.push(s[i]);
        }
    }
    int renum[6], totnum = num.size();
    while (!num.empty())
    {
        renum[totnum--] = num.top();
        num.pop();
    }
    char reop[6];
    int totop = op.size();
    int tot = totop;
    while (!op.empty())
    {
        reop[totop--] = op.top();
        op.pop();
    }
    for (int i = 1; i <= tot; i++)
    {
        if (reop[i] == '+')
            renum[i + 1] += renum[i];
        else
            renum[i + 1] = renum[i] - renum[i + 1];
    }
    //printf("%d\n",renum[tot+1]);
    if (renum[tot + 1] == 24)
        printf("Yes\n");
    else
        printf("No\n");
}
int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        work();
    }
}
//6*4+4/5